"""Research 节点:基于分析结果做 RAG 检索.

从 analysis_result 提取关键词,执行向量检索,写入 retrieved_docs.
"""
import re
import time

from .. import config
from ..state import LegalState
from ..vector_store import similarity_search_multi

# 单轮最多使用的关键词数量,防止用户在前端填入过多关键词导致多次检索
_MAX_KEYWORDS = 5


def _extract_keywords(analysis: str, question: str) -> list[str]:
    """从 analysis_result 的"关键词:"行解析多个关键词.

    兼容中英文逗号、顿号、分号和竖线分隔(LLM 或用户手填都可能混用),
    并去掉可能残留的引号/括号.解析不到任何关键词时回退到原始问题.
    """
    match = re.search(r"关键词[:：]\s*(.+)", analysis)
    if not match:
        return [question]

    parts = re.split(r"[,，、;；|]", match.group(1))
    keywords = [kw.strip(" \t\"'`“”‘’[]【】") for kw in parts]
    keywords = [kw for kw in keywords if kw]
    # 去重并限制数量(保持用户/LLM 给出的顺序)
    keywords = list(dict.fromkeys(keywords))[:_MAX_KEYWORDS]
    return keywords or [question]


def research_node(state: LegalState) -> dict:
    """检索节点:用 analysis 的全部关键词做多查询 RAG."""
    analysis = state.get("analysis_result", "")
    question = state["question"]
    keywords = _extract_keywords(analysis, question)

    t0 = time.time()
    docs = similarity_search_multi(keywords, k=config.RAG_TOP_K)
    print(
        f"[Research] 检索关键词: {keywords}, 命中 {len(docs)} 条,"
        f"耗时 {time.time()-t0:.2f}s"
    )

    return {"retrieved_docs": docs}
