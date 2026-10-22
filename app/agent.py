"""法律咨询 Agent 入口.

LangGraph 改造后,入口只负责:
1. 构造本轮输入 state(只含 question 和新 HumanMessage,历史由 checkpointer 加载)
2. 调用 LangGraph(astream_events 流式),通过 thread_id 绑定 session_id
3. 多轮记忆由 MemorySaver checkpointer 持久化,无需手动管理 history 列表

第 5 步:Interrupts 人工介入.分析确认节点会 interrupt 暂停,ask_stream 检测到
中断后 yield interrupt 事件给前端;用户确认后前端调 /resume,resume_stream 用
Command(resume=...) 恢复继续流式输出.
"""
import time
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from .graph import legal_graph
from .state import LegalState


# 节点名 → 用户可见的进度提示
_NODE_STATUS = {
    "supervisor": "正在理解问题...",
    "analysis": "正在分析法律问题...",
    "analysis_confirm": "等待确认分析结果...",
    "sql_node": "正在查询案例库...",
    "research": "正在检索法条...",
    "synthesis": "正在撰写回答...",
    "fast_rag": "正在检索并生成回答...",
    "parse": "正在整理回答...",
    "fallback": "正在整理回答...",
}

# 会触发 token 流式的节点(使用 llm.astream,fast_rag 用 invoke 不产生 token 事件)
_STREAM_NODES = {"synthesis"}


def _build_input_state(question: str) -> LegalState:
    """构造本轮输入 state.

    只传 question 和本轮新 HumanMessage;messages 用 add_messages reducer,
    checkpointer 会自动加载该 thread_id 上次的 messages 历史并追加本条.
    其他字段留给节点按需写入.
    """
    return {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "needs_sql": False,
        "retrieved_docs": [],
        "case_data": "",
        "final_answer": "",
        "error": None,
    }


class _StreamCtx:
    """流式事件累积上下文,供 ask_stream 和 resume_stream 共享.

    latest_final: 最后一次带 final_answer 的节点输出(首选)
    token_accum: synthesis 流式 token 累积(兜底)
    emitted_status: 已发过 status 的节点(去重)
    """
    def __init__(self):
        self.latest_final = ""
        self.token_accum = ""
        self.emitted_status: set[str] = set()


async def _consume_events(event: dict, ctx: _StreamCtx) -> dict | None:
    """处理单个 astream_events 事件,返回要 yield 的事件 dict 或 None.

    返回值:
      {"type":"status","content":...}  节点首次启动
      {"type":"token","content":...}  synthesis 流式 token
      None                            其他事件(忽略)
    注:final_answer 的捕获写入 ctx.latest_final,由调用方在流结束后统一处理.
    """
    etype = event["event"]
    meta = event.get("metadata") or {}
    node = meta.get("langgraph_node", "")

    # 1) 节点进度:每个节点首次启动时发一次
    if etype == "on_chain_start" and node in _NODE_STATUS and node not in ctx.emitted_status:
        ctx.emitted_status.add(node)
        return {"type": "status", "content": _NODE_STATUS[node]}

    # 2) token 流式:只转发 synthesis(已改为 astream)的 token
    if etype == "on_chat_model_stream" and node in _STREAM_NODES:
        chunk = (event.get("data") or {}).get("chunk")
        content = getattr(chunk, "content", "") if chunk is not None else ""
        if content:
            ctx.token_accum += content
            return {"type": "token", "content": content}

    # 3) 捕获最终回答:取最后一个带 final_answer 的节点输出
    #    (根图 on_chain_end 的 output 是完整最终状态,排在最后)
    if etype == "on_chain_end":
        out = (event.get("data") or {}).get("output")
        if isinstance(out, dict):
            fa = out.get("final_answer")
            if fa:  # 非空才更新(空 dict 透传不覆盖)
                ctx.latest_final = fa

    return None


async def _detect_interrupt(config: dict) -> dict | None:
    """检测图是否处于 interrupt 等待状态.

    astream 跑完后调 aget_state,若 snap.next 非空且有 task 的 interrupts 非空,
    返回 interrupt 载荷(interrupt.value);否则返回 None(本轮已完成).
    """
    snap = await legal_graph.aget_state(config)
    if not snap.next:
        return None
    for task in snap.tasks:
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0].value
    return None


def _final_answer(ctx: _StreamCtx) -> str:
    """确定最终回答:on_chain_end 捕获优先,token 累积兜底."""
    return ctx.latest_final or ctx.token_accum or "抱歉,无法处理您的问题,请稍后重试."


async def ask_stream(
    question: str, session_id: str
) -> AsyncGenerator[dict, None]:
    """流式处理用户问题,逐步 yield 事件 dict.

    多轮记忆:通过 thread_id=session_id 绑定 MemorySaver checkpointer,
    同一 session_id 的历史 messages 会自动加载并参与本轮上下文.

    事件类型:
      {"type": "status",    "content": "..."}   节点进度
      {"type": "token",     "content": "..."}   LLM token(synthesis 流式)
      {"type": "interrupt", "content": {...}, "thread_id": "..."}  分析确认中断,等待用户
      {"type": "done",      "content": "..."}    最终完整回答,流式结束信号
    """
    t0 = time.time()
    input_state = _build_input_state(question)
    run_config = {"configurable": {"thread_id": session_id}}

    ctx = _StreamCtx()
    async for event in legal_graph.astream_events(
        input_state, config=run_config, version="v2"
    ):
        out = await _consume_events(event, ctx)
        if out is not None:
            yield out

    # 图执行完毕:检测是否处于 interrupt 等待
    iv = await _detect_interrupt(run_config)
    if iv is not None:
        yield {"type": "interrupt", "content": iv, "thread_id": session_id}
        return  # 等待用户 resume,本轮不 yield done

    answer = _final_answer(ctx)
    total = time.time() - t0
    print(f"[Timing] === 总耗时(stream): {total:.1f}s ===")
    yield {"type": "done", "content": answer}


async def resume_stream(
    session_id: str, resume_value: dict
) -> AsyncGenerator[dict, None]:
    """恢复被 interrupt 暂停的图,继续流式输出.

    用 Command(resume=resume_value) 作为 astream_events 的 input 触发恢复,
    checkpointer 会从上次 interrupt 处继续执行剩余节点(sql/research/synthesis/parse).
    """
    t0 = time.time()
    run_config = {"configurable": {"thread_id": session_id}}
    cmd = Command(resume=resume_value)

    ctx = _StreamCtx()
    async for event in legal_graph.astream_events(
        cmd, config=run_config, version="v2"
    ):
        out = await _consume_events(event, ctx)
        if out is not None:
            yield out

    # 容错:理论上不会再中断,但检测一下
    iv = await _detect_interrupt(run_config)
    if iv is not None:
        yield {"type": "interrupt", "content": iv, "thread_id": session_id}
        return

    answer = _final_answer(ctx)
    total = time.time() - t0
    print(f"[Timing] === 总耗时(resume): {total:.1f}s ===")
    yield {"type": "done", "content": answer}
