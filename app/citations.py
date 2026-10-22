"""法条引用核验.

LLM 会凭记忆引用未出现在检索上下文中的条文,评测里记为"引用未落地".
这里在生成后逐条回查语料库:能查到同一法律、同一条文的 chunk,就补进
retrieved_docs(引用有据可查);查不到的只记录日志,不伪造依据.
"""
import os
import re

from .vector_store import query_by_article

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

_CITE_RE = re.compile(r"《([^》]{2,30})》\s*第([0-9一二三四五六七八九十百千零两]+)条")
_DATE_SUFFIX_RE = re.compile(r"_\d{8}$")

# 模型可能用简称,统一到语料库里的法律名再比对
_LAW_ALIASES = {
    "刑诉法": "刑事诉讼法",
    "民诉法": "民事诉讼法",
    "消保法": "消费者权益保护法",
    "道交法": "道路交通安全法",
}


def cn_to_int(text: str):
    """中文数字转整数,支持 1-9999;纯阿拉伯数字直接返回."""
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total, current = 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[ch]
            current = 0
        else:
            return None
    return total + current


def int_to_cn(num: int) -> str:
    """整数转中文数字(1-9999),用于匹配 metadata 里的条文号写法."""
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    if num == 0:
        return "零"
    s = ""
    text = str(num)
    zero = False
    for i, ch in enumerate(text):
        d = int(ch)
        pos = len(text) - i - 1
        if d == 0:
            zero = True
            continue
        if zero and s:
            s += "零"
        zero = False
        if d == 1 and pos == 1 and not s:
            s += "十"
        else:
            s += digits[d] + units[pos]
    return s


def _norm_law(name: str) -> str:
    name = name.replace("中华人民共和国", "")
    name = re.sub(r"[《》\s]", "", name)
    return _LAW_ALIASES.get(name, name)


def _law_from_source(source: str) -> str:
    stem = source.rsplit(".", 1)[0] if source else ""
    return _norm_law(_DATE_SUFFIX_RE.sub("", stem))


def extract_citations(text: str) -> list[tuple[str, int]]:
    """抽取《法名》第X条形式的引用."""
    out = []
    for name, num in _CITE_RE.findall(text or ""):
        n = cn_to_int(num)
        if n:
            out.append((_norm_law(name), n))
    return out


def _grounded(law: str, n: int, docs) -> bool:
    """同一 chunk 的 metadata.article 是该条文,且来源法律名一致."""
    variants = {f"第{n}条", f"第{int_to_cn(n)}条"}
    for doc in docs:
        article = re.sub(r"\s", "", (doc.metadata or {}).get("article", "") or "")
        if article not in variants:
            continue
        if _law_from_source((doc.metadata or {}).get("source", "")) == law:
            return True
    return False


def verify_and_attach(answer: str, docs: list) -> list:
    """核验答案中的每条引用,并把缺失的条文原文补进检索结果.

    只补语料库里真实存在的同法同条 chunk;查不到的记入日志,不做任何伪造.
    """
    # 评测对照用:关掉核验,直接测检索本身能落地多少引用
    if os.getenv("CITATION_VERIFY") == "0":
        return docs

    cites = extract_citations(answer)
    if not cites:
        return docs

    docs = list(docs)
    verified, missing = 0, []
    by_article: dict[int, list] = {}

    for law, n in cites:
        if _grounded(law, n, docs):
            verified += 1
            continue
        if n not in by_article:
            by_article[n] = query_by_article([f"第{n}条", f"第{int_to_cn(n)}条"])
        match = next(
            (d for d in by_article[n] if _law_from_source(d.metadata["source"]) == law),
            None,
        )
        if match is None:
            missing.append(f"{law}第{n}条")
            continue
        docs.append(match)
        verified += 1

    print(
        f"[Citation] 引用核验: 共 {len(cites)} 条, 已落地 {verified} 条, "
        f"库中未找到 {len(missing)} 条{(' ' + ', '.join(missing)) if missing else ''}"
    )
    return docs
