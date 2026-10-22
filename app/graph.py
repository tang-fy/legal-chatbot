"""LangGraph 图构建.

第 3 步:Agent 分支并行优化(sql_node ∥ research)
第 4 步:加 MemorySaver checkpointer,支持基于 thread_id 的多轮记忆

图结构:
  START → supervisor → (条件路由)
    ├─ needs_sql=False → fast_rag → parse → END
    └─ needs_sql=True  → analysis ─┬→ sql_node ──┐
                                    └→ research ──┴→ synthesis → parse → END
                       (fan-out)              (fan-in 屏障:等两者都完成)

checkpointer:
  使用 MemorySaver(内存级,进程内单例),按 thread_id 持久化 LegalState
  (含 messages 列表),同一 thread_id 的下次调用会自动加载上次最终 state,
  实现多轮记忆.生产环境可换 RedisSaver/PostgresSaver.
"""
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import LegalState
from .nodes import (
    supervisor_node,
    fast_rag_node,
    analysis_node,
    analysis_confirm_node,
    research_node,
    sql_node,
    synthesis_node,
    parse_node,
    fallback_node,
)


def _route_after_supervisor(state: LegalState) -> Literal["fast_rag", "analysis"]:
    """条件路由:needs_sql=False 走 fast_rag,True 走 analysis."""
    if state.get("needs_sql"):
        return "analysis"
    return "fast_rag"


def _route_after_parse(state: LegalState) -> Literal["fallback", "__end__"]:
    """解析成功 → END;失败 → fallback."""
    if state.get("error") == "parse_failed":
        return "fallback"
    return END


def build_graph():
    """构建并编译 LangGraph 图(带 MemorySaver checkpointer).

    Agent 分支采用 fan-out/fan-in 并行模式:
      analysis 同时派发 sql_node 与 research(LangGraph 自动并行执行),
      synthesis 作为 fan-in 屏障,等待两者都完成后才运行.
    两个节点写不同字段(case_data / retrieved_docs),无写冲突,无需额外 reducer.

    checkpointer=MemorySaver():
      调用 legal_graph 时需传 config={"configurable": {"thread_id": session_id}},
      同一 thread_id 会自动加载上次最终 state(含 messages 历史),
      节点只需 update messages 字段,持久化由 checkpointer 接管.
    """
    graph = StateGraph(LegalState)

    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("fast_rag", fast_rag_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("analysis_confirm", analysis_confirm_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("parse", parse_node)
    graph.add_node("fallback", fallback_node)

    # 入口 → supervisor
    graph.add_edge(START, "supervisor")

    # supervisor → 条件路由(Fast Path / Agent 分支)
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "fast_rag": "fast_rag",
            "analysis": "analysis",
        },
    )

    # === Fast Path 分支 ===
    graph.add_edge("fast_rag", "parse")

    # === Agent 分支:fan-out / fan-in 并行 ===
    # 第 5 步:analysis 先产出草稿 → analysis_confirm(interrupt 暂停等用户确认)
    # → 确认后才 fan-out 到 sql_node 和 research
    graph.add_edge("analysis", "analysis_confirm")
    # analysis_confirm 同时派发 sql_node 和 research(并行执行)
    graph.add_edge("analysis_confirm", "sql_node")
    graph.add_edge("analysis_confirm", "research")
    # synthesis 作为 fan-in 屏障:等 sql_node 和 research 都完成
    graph.add_edge("sql_node", "synthesis")
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", "parse")

    # === 共享后置:parse → 条件路由 ===
    graph.add_conditional_edges(
        "parse",
        _route_after_parse,
        {
            "fallback": "fallback",
            END: END,
        },
    )

    # fallback → END
    graph.add_edge("fallback", END)

    # 带 MemorySaver checkpointer 编译,支持基于 thread_id 的多轮记忆持久化
    return graph.compile(checkpointer=MemorySaver())


# 模块级单例:整个应用共享一个编译好的 graph
legal_graph = build_graph()
