"""解析节点:把 LLM 输出解析为五段式结构化回答.

解析失败时走 fallback_node.

多轮记忆:解析成功后把 AIMessage(final_answer) 写回 messages,
由 MemorySaver checkpointer 持久化,供下一轮作为历史上下文使用.
"""
import re
from typing import Optional

from langchain_core.messages import AIMessage

from ..output_parser import output_parser, LegalAnswerModel


def _parse_legal_answer(output: str) -> Optional[LegalAnswerModel]:
    """解析模型输出为 LegalAnswerModel,失败返回 None."""
    if not output:
        return None
    try:
        return output_parser.parse(output)
    except Exception:
        pass
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            return output_parser.parse(match.group(0))
        except Exception:
            pass
    return None


def parse_node(state) -> dict:
    """解析节点:把 final_answer 解析/规整为五段式.

    - 若为 JSON(旧格式/fast_rag 输出):解析为 LegalAnswerModel 并格式化五段.
    - 若已是五段式纯文本(synthesis 流式输出):直接透传,不报错.
    - 若为空:标记 error,路由到 fallback_node 给出兜底提示.
    """
    raw = state.get("final_answer", "")
    parsed = _parse_legal_answer(raw)

    if parsed is not None:
        # JSON → 五段式格式化
        answer = (
            f"行为分析:\n{parsed.behavior_analysis}\n\n"
            f"法律后果:\n{parsed.consequences}\n\n"
            "行动建议:\n" + parsed.advice + "\n\n"
            f"法律依据:\n{parsed.legal_basis}\n\n"
            f"免责声明:\n{parsed.disclaimer}\n\n"
        )
        # 写回 AIMessage,让 checkpointer 持久化助手回复(多轮记忆)
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    # 非 JSON:若已有内容(纯文本)直接透传;若为空才走兜底
    if not raw.strip():
        print("[Warn] final_answer 为空,走兜底节点")
        return {"error": "parse_failed"}
    # 已是格式化文本,透传即可;同样写回 AIMessage 供多轮记忆
    return {"messages": [AIMessage(content=raw)]}


def fallback_node(state) -> dict:
    """兜底节点:final_answer 为空时给出默认提示,否则保留原值."""
    if not state.get("final_answer", "").strip():
        answer = "抱歉,生成回答时出现问题,请稍后重试或换一种方式描述您的问题."
        return {
            "final_answer": answer,
            "error": None,
            "messages": [AIMessage(content=answer)],
        }
    # 保留 final_answer 原值,清掉 error 标记;补写 AIMessage(parse 路径未写)
    return {
        "error": None,
        "messages": [AIMessage(content=state["final_answer"])],
    }
