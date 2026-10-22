"""LangGraph 共享状态定义.

所有节点共享此 State,通过读取/写入字段协作.
"""
from typing import TypedDict, Optional, List, Annotated

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class LegalState(TypedDict):
    """法律咨询 Agent 的共享状态."""
    # 输入:用户原始问题
    question: str
    # 多轮对话历史(reducer=add_messages,自动追加+截断)
    messages: Annotated[List[BaseMessage], add_messages]
    # 路由决策
    needs_sql: bool
    # analysis 节点输出:问题分析结果(法律领域/检索关键词)
    analysis_result: str
    # analysis 节点产出的结构化草稿,供 analysis_confirm 节点 interrupt 给用户确认
    analysis_draft: dict
    # RAG 检索结果
    retrieved_docs: List[Document]
    # NL2SQL 查询结果
    case_data: str
    # 最终回答
    final_answer: str
    # 错误兜底
    error: Optional[str]
