"""Analysis 节点:拆解用户问题.

第 5 步:拆成 analysis(跑 LLM 生成结构化草稿) + analysis_confirm(interrupt 暂停等用户确认/修改).
拆分原因:LangGraph 的 interrupt 恢复会重跑整个节点,若 interrupt 与 LLM 调用同处一节点,
恢复时会再调一次 DeepSeek(耗时+费钱).拆分后 interrupt 只重跑廉价的 confirm 节点.
"""
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.types import interrupt

from ..llm import llm

# 分析结果的结构化 Schema(轻量,7B/DeepSeek 都能稳定输出)
class AnalysisResult(BaseModel):
    """问题分析结果."""
    legal_domain: str = Field(description="法律领域:民事/刑事/行政/其他")
    brief_analysis: str = Field(description="问题摘要,一句话")
    search_keywords: list[str] = Field(description="用于检索法律条文的2-3个关键词")

_analysis_prompt = """你是一位法律问题分析师.任务:分析用户问题,提取关键信息供后续检索使用.

要求:
1. legal_domain: 判断属于民事/刑事/行政/其他哪一类
2. brief_analysis: 用一句话概括用户在问什么
3. search_keywords: 给出 2-3 个适合向量检索的关键词(法律术语优先,如"交通肇事罪""合同违约""工伤认定")

用户问题:
{question}

请输出 JSON:"""


def analysis_node(state) -> dict:
    """分析节点:用结构化输出拆解问题,结果存入 analysis_draft(不直接拼字符串)."""
    question = state["question"]

    # 用 with_structured_output 保证输出严格符合 Schema
    structured_llm = llm.with_structured_output(AnalysisResult)
    messages = [
        SystemMessage(content=_analysis_prompt.format(question=question)),
        HumanMessage(content=question),
    ]

    try:
        result = structured_llm.invoke(messages)
        print(f"[Analysis] 领域={result.legal_domain} 摘要={result.brief_analysis} 关键词={result.search_keywords}")
        return {"analysis_draft": {
            "legal_domain": result.legal_domain,
            "brief_analysis": result.brief_analysis,
            "search_keywords": list(result.search_keywords),
        }}
    except Exception as e:
        # 兜底:分析失败,用原始问题作为检索关键词
        print(f"[Analysis] 失败,用原始问题兜底: {e}")
        return {"analysis_draft": {
            "legal_domain": "未知",
            "brief_analysis": question,
            "search_keywords": [question],
        }}


def analysis_confirm_node(state) -> dict:
    """确认节点:interrupt 把分析草稿推给前端,用户确认/修改后再拼装 analysis_result.

    interrupt(value) 首次调用暂停图;恢复时本节点从头重跑,interrupt() 返回
    Command(resume=...) 传入的 resume_value.
    resume_value schema:
      {"action":"approve"}  接受 LLM 分析
      {"action":"edit","legal_domain":"...","brief_analysis":"...","search_keywords":[...]}  修改后确认
    对未知 action 一律按 approve 处理,防止前端传错 payload 卡死流程.
    """
    d = state.get("analysis_draft") or {}
    rv = interrupt({
        "legal_domain": d.get("legal_domain", ""),
        "brief_analysis": d.get("brief_analysis", ""),
        "search_keywords": d.get("search_keywords", []),
    }) or {}

    if rv.get("action") == "edit":
        domain = rv.get("legal_domain") or d.get("legal_domain", "")
        brief = rv.get("brief_analysis") or d.get("brief_analysis", "")
        kws = rv.get("search_keywords") or d.get("search_keywords", [])
        print(f"[AnalysisConfirm] 用户修改: 领域={domain} 摘要={brief} 关键词={kws}")
    else:
        domain = d.get("legal_domain", "")
        brief = d.get("brief_analysis", "")
        kws = d.get("search_keywords", [])
        print(f"[AnalysisConfirm] 用户接受原始分析")

    analysis_text = f"领域:{domain}\n摘要:{brief}\n关键词:{', '.join(kws)}"
    return {"analysis_result": analysis_text}
