"""法律问答评测脚本(骨架).

用法(在项目根目录执行):
  python eval/run_eval.py --suite route
  python eval/run_eval.py --suite retrieval --ks 3 5 --thresholds 0.5 0.8 1.0 999
  python eval/run_eval.py --suite e2e --limit 10
  python eval/run_eval.py --suite all --limit 10

三个子集:
  route     路由评测:直接调 supervisor_node,零 LLM 成本
  retrieval 检索参数评测:Top-K / L2 阈值,只消耗 embedding
  e2e       端到端评测:真实跑图,统计延迟/调用数/引用/拒答

产物: eval/out/ 下的 details.csv 与 summary.md
"""
import argparse
import asyncio
import csv
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 直接以 eval/run_eval.py 运行时,sys.path[0] 是 eval/,需把项目根目录加入才能 import app.*
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "eval" / "out"

# 输出文件名前缀(如 holdout_),避免 holdout 结果覆盖 60 题主集结果
_OUT_PREFIX = ""

# ===== 法条文本处理 =====

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

_CITE_RE = re.compile(r"《([^》]{2,30})》\s*第([0-9一二三四五六七八九十百千零两]+)条")
_ARTICLE_RE = re.compile(r"第([0-9一二三四五六七八九十百千零两]+)条")


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
    """整数转中文数字(1-9999),用于生成条文号的另一种写法."""
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


def norm_law(name: str) -> str:
    """归一化法律名称:去《》、空格与"中华人民共和国"前缀."""
    name = name.replace("中华人民共和国", "")
    return re.sub(r"[《》\s]", "", name)


def law_variants(law: str) -> list[str]:
    """生成条文号的多种写法,用于在检索文档中做文本匹配."""
    m = _ARTICLE_RE.search(law)
    if not m:
        return [norm_law(law)]
    num = cn_to_int(m.group(1))
    if num is None:
        return [norm_law(law)]
    name = norm_law(law[: m.start()])
    variants = [f"第{num}条", f"第{int_to_cn(num)}条"]
    if name:
        variants += [name + variants[0], name + variants[1]]
    return variants


def docs_text(docs) -> str:
    """把检索结果拼成无空白文本,便于条文号匹配."""
    return re.sub(r"\s", "", "".join(getattr(d, "page_content", "") for d in docs))


# 模型可能使用简称,统一到文档里的法律名后再比对
_LAW_ALIASES = {
    "刑诉法": "刑事诉讼法",
    "民诉法": "民事诉讼法",
    "消保法": "消费者权益保护法",
    "道交法": "道路交通安全法",
}
_SOURCE_DATE_RE = re.compile(r"_\d{8}$")


def _law_from_source(source: str) -> str:
    """由 chunk 的文件名得到法律名,如 中华人民共和国刑法_20201226.docx -> 刑法."""
    stem = os.path.splitext(source or "")[0]
    return norm_law(_SOURCE_DATE_RE.sub("", stem))


def _norm_law_name(law: str) -> str:
    name = norm_law(law)
    return _LAW_ALIASES.get(name, name)


def citation_grounded(law: str, n: int, docs, strict: bool = True) -> bool:
    """判断引用是否落在检索上下文里.

    strict=True: 同一 chunk 的 metadata.article 必须等于该条文号,且来源法律名匹配,
                 避免"公司法第266条"被算作"刑法第266条"的落地引用.
    strict=False: 只要任一检索 chunk 出现该条文号(旧口径,仅用于与历史数据对比).
    """
    variants = {f"第{n}条", f"第{int_to_cn(n)}条"}
    target = _norm_law_name(law)
    for doc in docs:
        text = re.sub(r"\s", "", doc.page_content)
        article = re.sub(r"\s", "", (doc.metadata or {}).get("article", "") or "")
        hit = article in variants if article else any(v in text for v in variants)
        if not hit:
            continue
        if not strict:
            return True
        if _law_from_source((doc.metadata or {}).get("source", "")) == target:
            return True
    return False


def law_in_docs(law: str, text: str) -> bool:
    return any(v in text for v in law_variants(law))


def parse_expected(law: str):
    """解析期望法条为 (法律名, 条号)."""
    m = _ARTICLE_RE.search(law)
    if not m:
        return (norm_law(law), None)
    return (norm_law(law[: m.start()]), cn_to_int(m.group(1)))


def extract_citations(answer: str) -> list[tuple[str, int]]:
    """从答案中抽取《法名》第X条形式的引用."""
    out = []
    for name, num in _CITE_RE.findall(answer or ""):
        n = cn_to_int(num)
        if n:
            out.append((norm_law(name), n))
    return out


_NUM_UNITS = r"年|日|天|个月|月|倍|次|人|元|岁|周|小时|分钟|件|起|宗|户|%"
_CN_NUM_RE = re.compile(rf"[零一二三四五六七八九十百两]+(?={_NUM_UNITS})")
_PERCENT_CN_RE = re.compile(r"百分之([零一二三四五六七八九十百两]+)")
_DATE_RE = re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")


def normalize_numbers(text: str) -> str:
    """把中文数字与阿拉伯数字、日/天写法归一,避免"三年" vs "3年"误判为未命中."""
    text = (text or "").replace("％", "%")

    def _percent(m):
        n = cn_to_int(m.group(1))
        return f"{n}%" if n is not None else m.group(0)

    text = _PERCENT_CN_RE.sub(_percent, text)

    def _num(m):
        n = cn_to_int(m.group(0))
        return str(n) if n is not None else m.group(0)

    text = _CN_NUM_RE.sub(_num, text)

    # 先把日期换成占位符,避免下面的"日→天"把日期末尾的"日"也改掉
    dates = []

    def _stash_date(m):
        dates.append(f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日")
        return f"\x00{len(dates) - 1}\x00"

    text = _DATE_RE.sub(_stash_date, text)
    # "30日" 与 "30天" 视为同一表述
    text = re.sub(r"(\d+)日", r"\1天", text)
    # 还原日期:"2023-05-10" 与 "2023年5月10日" 归一
    return re.sub(r"\x00(\d+)\x00", lambda m: dates[int(m.group(1))], text)


def keyword_hit_rate(answer: str, keywords: list[str]):
    """答案对必答要点的覆盖率;无标注时返回 None."""
    if not keywords:
        return None
    text = normalize_numbers(re.sub(r"\s", "", answer or ""))
    hits = sum(
        1 for k in keywords if normalize_numbers(re.sub(r"\s", "", k)) in text
    )
    return hits / len(keywords)


_REFUSE_MARKERS = ("未找到", "未检索到", "无法基于现有", "无法回答", "超出", "不属于")


def is_refusal(answer: str) -> bool:
    """判断答案是否为明确拒答(含拒答话术且未编造具体条文)."""
    return any(m in (answer or "") for m in _REFUSE_MARKERS) and not extract_citations(answer)


# ===== 通用 IO =====

def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"questions.jsonl 第 {line_no} 行 JSON 解析失败: {e}")
    return rows


def write_csv(name: str, rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / (_OUT_PREFIX + name)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(name: str, text: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / (_OUT_PREFIX + name)
    path.write_text(text, encoding="utf-8")
    return path


def pct(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def percentile(values: list[float], p: float):
    """线性插值分位数,p 取 0-100."""
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def preflight_milvus():
    """检索前快速检查 Milvus 是否可达,避免 30 次重试等 150 秒."""
    from app import config
    host = config.MILVUS_HOST
    if host == "milvus":
        raise SystemExit(
            "MILVUS_HOST 仍是容器内默认值 'milvus'. 本地评测请先设置 MILVUS_HOST=localhost"
        )
    try:
        with socket.create_connection((host, int(config.MILVUS_PORT)), timeout=3):
            return
    except OSError as e:
        raise SystemExit(f"无法连接 Milvus {host}:{config.MILVUS_PORT} ({e}), 请先启动 Docker 基础设施")


# ===== E1 路由评测 =====

def run_route(questions: list[dict]) -> dict:
    from app.nodes.supervisor import supervisor_node

    rows = []
    tp = fp = tn = fn = 0
    for q in questions:
        actual = "agent" if supervisor_node({"question": q["question"]})["needs_sql"] else "fast"
        expected = q["expected_route"]
        if expected == "agent" and actual == "agent":
            tp += 1
        elif expected == "agent" and actual == "fast":
            fn += 1
        elif expected == "fast" and actual == "agent":
            fp += 1
        else:
            tn += 1
        rows.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "expected_route": expected,
            "actual_route": actual,
            "correct": expected == actual,
            "question": q["question"],
        })

    n = len(rows)
    accuracy = (tp + tn) / n if n else None
    agent_recall = tp / (tp + fn) if (tp + fn) else None
    agent_precision = tp / (tp + fp) if (tp + fp) else None
    fast_fp_rate = fp / (fp + tn) if (fp + tn) else None

    path = write_csv("route_details.csv", rows)
    errors = [r for r in rows if not r["correct"]]
    lines = [
        "# 路由评测",
        "",
        f"- 题数: {n}",
        f"- 准确率: {pct(accuracy)}",
        f"- Agent 类召回: {pct(agent_recall)}",
        f"- Agent 类精确率: {pct(agent_precision)}",
        f"- Fast 误判为 Agent 的比例: {pct(fast_fp_rate)}",
        f"- 明细: `{path.name}`",
        "",
        "## 误判样本",
        "",
        "| id | 期望 | 实际 | 问题 |",
        "|---|---|---|---|",
    ]
    for r in errors:
        lines.append(f"| {r['id']} | {r['expected_route']} | {r['actual_route']} | {r['question']} |")
    summary = write_summary("route_summary.md", "\n".join(lines) + "\n")

    print(f"[Route] 准确率 {pct(accuracy)}, Agent 召回 {pct(agent_recall)}, 明细 {path}")
    print(f"[Route] 汇总 {summary}")
    return {"accuracy": accuracy, "agent_recall": agent_recall}


# ===== E3 检索评测 =====

def run_retrieval(questions: list[dict], ks: list[int], thresholds: list[float]) -> dict:
    from app.vector_store import similarity_search

    preflight_milvus()
    items = [q for q in questions if q.get("expected_laws") and q.get("retrieval_query")]
    if not items:
        raise SystemExit("没有带 expected_laws + retrieval_query 的题目,无法做检索评测")

    rows = []
    matrix = []
    for k in ks:
        for th in thresholds:
            hit = 0
            source_hit = 0
            irrelevant_rates = []
            for q in items:
                docs = similarity_search(q["retrieval_query"], k=k, threshold=th)
                text = docs_text(docs)
                matched = any(law_in_docs(law, text) for law in q["expected_laws"])
                src_matched = any(law_source_hit(law, d) for law in q["expected_laws"] for d in docs)
                hit += 1 if matched else 0
                source_hit += 1 if src_matched else 0
                if docs:
                    # 来源级无关:文档不是出自任何期望的法律文件(同法不同条不算无关)
                    bad = sum(
                        1 for d in docs
                        if not any(law_source_hit(law, d) for law in q["expected_laws"])
                    )
                    irrelevant_rates.append(bad / len(docs))
                else:
                    irrelevant_rates.append(1.0)
                rows.append({
                    "id": q["id"],
                    "k": k,
                    "threshold": th,
                    "returned": len(docs),
                    "article_matched": matched,
                    "source_matched": src_matched,
                    "sources": ";".join(d.metadata.get("source", "") for d in docs),
                    "retrieval_query": q["retrieval_query"],
                })
            recall = hit / len(items)
            source_recall = source_hit / len(items)
            irrelevant = sum(irrelevant_rates) / len(irrelevant_rates) if irrelevant_rates else None
            matrix.append({
                "k": k, "threshold": th, "recall": recall,
                "source_recall": source_recall, "irrelevant": irrelevant,
            })

    path = write_csv("retrieval_details.csv", rows)
    lines = [
        "# 检索评测",
        "",
        f"- 题数(带法条标注): {len(items)}",
        f"- 明细: `{path.name}`",
        "",
        "| Top-K | L2 阈值 | 条文级 recall@K | 来源级 recall@K | 来源无关文档占比 |",
        "|---|---|---|---|---|",
    ]
    for m in matrix:
        lines.append(
            f"| {m['k']} | {m['threshold']} | {pct(m['recall'])} | "
            f"{pct(m['source_recall'])} | {pct(m['irrelevant'])} |"
        )
    summary = write_summary("retrieval_summary.md", "\n".join(lines) + "\n")

    print(f"[Retrieval] 完成 {len(items)} 题 × {len(matrix)} 组参数, 明细 {path}")
    print(f"[Retrieval] 汇总 {summary}")
    return {"items": len(items), "matrix": matrix}


def law_in_docs_any(laws: list[str], doc) -> bool:
    text = re.sub(r"\s", "", getattr(doc, "page_content", ""))
    return any(law_in_docs(law, text) for law in laws)


def law_source_hit(law: str, doc) -> bool:
    """判断文档来源文件是否属于期望的那部法律(同法不同条也算来源命中)."""
    name = re.split(r"第[0-9一二三四五六七八九十百千零两]+条", law)[0].strip("《》 ")
    source = getattr(doc, "metadata", {}).get("source", "")
    return bool(name) and name in source


# ===== E2 端到端评测 =====

async def run_e2e(questions: list[dict], limit: int) -> dict:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from app.graph import legal_graph

    class LLMCounter(BaseCallbackHandler):
        def __init__(self):
            self.n = 0

        def on_chat_model_start(self, *args, **kwargs):
            self.n += 1

    rows = []
    selected = questions[:limit] if limit else questions
    for q in selected:
        counter = LLMCounter()
        cfg = {
            "configurable": {"thread_id": f"eval-{q['id']}-{int(time.time() * 1000)}"},
            "callbacks": [counter],
        }
        state = {
            "question": q["question"],
            "messages": [HumanMessage(content=q["question"])],
            "needs_sql": False,
            "retrieved_docs": [],
            "case_data": "",
            "final_answer": "",
            "error": None,
        }

        t0 = time.perf_counter()
        try:
            result = await legal_graph.ainvoke(state, cfg)
        except Exception:
            # 部分 LangGraph 版本在 interrupt 时抛异常;无待恢复任务则说明是真错误
            snap = await legal_graph.aget_state(cfg)
            if not snap.next:
                raise
            result = snap.values
        latency = time.perf_counter() - t0

        snap = await legal_graph.aget_state(cfg)
        if snap.next:  # 停在 analysis_confirm,自动确认后继续
            result = await legal_graph.ainvoke(
                Command(resume={"action": "approve"}), cfg
            )
            latency = time.perf_counter() - t0

        answer = (result or {}).get("final_answer") or ""
        docs = (result or {}).get("retrieved_docs") or []
        actual_route = "agent" if (result or {}).get("analysis_result") else "fast"

        expected_set = {parse_expected(law) for law in q.get("expected_laws", [])}
        cites = extract_citations(answer)
        # 严格口径:同一 law+article 的 chunk 才算落地;宽松口径仅用于和历史数据对比
        grounded = [(law, n) for (law, n) in cites if citation_grounded(law, n, docs)]
        grounded_loose = [
            (law, n) for (law, n) in cites if citation_grounded(law, n, docs, strict=False)
        ]
        cited_expected = {(law, n) for (law, n) in cites if (law, n) in expected_set}
        cit_hallucination = 1 - len(grounded) / len(cites) if cites else None
        cit_hallucination_loose = 1 - len(grounded_loose) / len(cites) if cites else None
        cit_recall = len(cited_expected) / len(expected_set) if expected_set else None
        kw_hit = keyword_hit_rate(answer, q.get("answer_keywords", []))

        refusal_ok = None
        if q.get("expected_behavior") == "refuse":
            refusal_ok = is_refusal(answer)

        rows.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "expected_route": q.get("expected_route", ""),
            "actual_route": actual_route,
            "route_correct": q.get("expected_route") == actual_route,
            "latency_s": round(latency, 3),
            "llm_calls": counter.n,
            "keyword_hit": None if kw_hit is None else round(kw_hit, 3),
            "citations": len(cites),
            "citation_hallucination": (
                None if cit_hallucination is None else round(cit_hallucination, 3)
            ),
            "citation_hallucination_loose": (
                None if cit_hallucination_loose is None else round(cit_hallucination_loose, 3)
            ),
            "citation_recall": None if cit_recall is None else round(cit_recall, 3),
            "refusal_ok": refusal_ok,
            "retrieved": len(docs),
            "answer": answer[:500],
            "question": q["question"],
        })

    path = write_csv("e2e_details.csv", rows)

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    latencies = [r["latency_s"] for r in rows]
    calls = [r["llm_calls"] for r in rows]
    route_acc = sum(1 for r in rows if r["route_correct"]) / len(rows)
    refusal_rows = [r for r in rows if r["refusal_ok"] is not None]
    refusal_rate = (
        sum(1 for r in refusal_rows if r["refusal_ok"]) / len(refusal_rows)
        if refusal_rows else None
    )

    lines = [
        "# 端到端评测",
        "",
        f"- 题数: {len(rows)}",
        f"- 路由准确率: {pct(route_acc)}",
        f"- 延迟 P50 / P95: {percentile(latencies, 50):.2f}s / {percentile(latencies, 95):.2f}s",
        f"- LLM 调用数 平均: {sum(calls) / len(calls):.2f}",
        f"- 要点命中率 平均: {pct(avg('keyword_hit'))}",
        f"- 引用未落地率(严格口径) 平均: {pct(avg('citation_hallucination'))}",
        f"- 引用未落地率(宽松口径,与历史数据同口径): {pct(avg('citation_hallucination_loose'))}",
        f"- gold 引用召回率 平均: {pct(avg('citation_recall'))}",
        f"- 正确拒答率: {pct(refusal_rate)}",
        f"- 明细: `{path.name}`",
        "",
        "## 需要人工复核的样本",
        "",
        "| id | 路由 | 要点命中 | 引用未落地率 | 问题 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if (
            (r["keyword_hit"] is not None and r["keyword_hit"] < 0.6)
            or r["refusal_ok"] is False
            or (r["citation_hallucination"] or 0) > 0
        ):
            lines.append(
                f"| {r['id']} | {r['actual_route']} | {pct(r['keyword_hit'])} | "
                f"{pct(r['citation_hallucination'])} | {r['question']} |"
            )
    summary = write_summary("e2e_summary.md", "\n".join(lines) + "\n")

    print(f"[E2E] {len(rows)} 题完成, P50 {percentile(latencies, 50):.2f}s, 明细 {path}")
    print(f"[E2E] 汇总 {summary}")
    return {"rows": len(rows), "p50": percentile(latencies, 50)}


# ===== 入口 =====

def parse_args():
    parser = argparse.ArgumentParser(description="法律问答评测")
    parser.add_argument("--suite", choices=["route", "retrieval", "e2e", "all"], default="route")
    parser.add_argument("--limit", type=int, default=0, help="e2e 只跑前 N 题,0 表示全部")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH),
                        help="题目文件,默认 eval/questions.jsonl;holdout 用 eval/holdout.jsonl")
    parser.add_argument("--prefix", default="",
                        help="输出文件名前缀,如 holdout_ (避免覆盖主集结果)")
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8, 1.0, 999.0])
    return parser.parse_args()


def main():
    global _OUT_PREFIX
    args = parse_args()
    _OUT_PREFIX = args.prefix
    # route / retrieval 不调用 LLM,但导入 app.nodes 会初始化 ChatOpenAI,
    # 缺少 key 时给一个占位值,避免仅评测路由也要配置真实密钥
    if args.suite in ("route", "retrieval") and not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-eval-offline"
        print("[Eval] 未设置 DEEPSEEK_API_KEY,route/retrieval 使用占位值(不调 LLM)")

    questions = load_questions(Path(args.questions))
    print(f"[Eval] 载入 {len(questions)} 题: {args.questions}")

    if args.suite in ("route", "all"):
        run_route(questions)
    if args.suite in ("retrieval", "all"):
        run_retrieval(questions, args.ks, args.thresholds)
    if args.suite in ("e2e", "all"):
        asyncio.run(run_e2e(questions, args.limit))


if __name__ == "__main__":
    main()
