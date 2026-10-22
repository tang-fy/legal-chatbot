"""Synthesis 节点:综合分析+检索+SQL结果,生成五段式回答.

第 3 步:改为 async + llm.astream(),支持 token 级流式输出.
输出格式由 JSON 改为五段式纯文本,使流式 token 直接可读(打字机效果),
parse_node 对非 JSON 输出做透传处理.
"""
import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .. import config
from ..citations import verify_and_attach
from ..llm import llm
from ..state import LegalState


def _format_history(messages) -> str:
    """把多轮历史拼成可读文本(本轮 HumanMessage 已作为 question 单独传入,这里排除最后一条).

    只取最近 HISTORY_LIMIT 轮(每轮=Human+AI),避免上下文过长.
    """
    if not messages:
        return ""
    # 最后一条是本轮 HumanMessage(question),历史取其之前的
    history_msgs = list(messages[:-1]) if len(messages) > 1 else []
    if not history_msgs:
        return ""
    limit = config.HISTORY_LIMIT * 2
    if len(history_msgs) > limit:
        history_msgs = history_msgs[-limit:]
    lines = []
    for m in history_msgs:
        role = "用户" if isinstance(m, HumanMessage) else ("律师" if isinstance(m, AIMessage) else "系统")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        lines.append(f"{role}: {content}")
    return "\n[对话历史]\n" + "\n".join(lines) + "\n\n"

# Synthesis 节点系统提示词(基于原 agent.py 的 system_prompt,
# 移除 JSON 格式说明,改为五段式纯文本输出以支持流式展示)
_synthesis_prompt = """你是一位经验丰富、善于向普通人解释法律问题的律师顾问。你的目标不是给用户罗列法条,而是像真人律师那样,针对用户描述的具体行为或情境,告诉TA:行为性质、是否违法、不同做法的不同后果、应如何合法应对。

**回答风格(非常重要):**
1. 开头先用一句话定性用户行为(如"您描述的'开车撞死行人'涉嫌交通肇事罪"),不要先介绍检索到的文档.
2. 结合案情中的具体当事人和具体行为逐一分析,说人话,严禁整段抄录或罗列法条原文.
3. **法律后果必须按情节轻重分情形说明**(如未逃逸/逃逸/逃逸致人死亡的不同刑期).
4. **必须给出可执行的合法做法**(如停车、报警、抢救、不要逃逸),让用户看完就知道怎么做.
5. 法律依据只引用 1-3 条最核心条文:法律名称+条款号+一句话概括.
6. 引用法律时注意:原《合同法》已废止,合同相关问题应引用《民法典》合同编.
7. 对于不确定的法律问题,提示用户咨询专业律师.
8. 若检索到的条文与用户问题不相关,不要照抄,基于自己的法律知识回答,并注明"相关条文未在检索结果中命中,依据一般法律知识".

**输出格式(必须严格遵守):**
直接输出以下五个段落,每段以"标题:"开头,段间用一个空行分隔.
不要输出 JSON、代码块或任何额外说明,只输出正文本身:

行为分析:
<对用户行为的法律性质、是否违法、构成要点的分析>

法律后果:
<按情节轻重分情形说明可能承担的民事/行政/刑事责任>

行动建议:
<可执行的具体合法做法>

法律依据:
<1-3 条核心条文:法律名+条款号+一句话概括>

免责声明:
本回答仅供参考,不构成正式法律意见,如需专业判断请咨询律师.
"""


async def synthesis_node(state: LegalState) -> dict:
    """综合节点:async + astream,流式生成五段式纯文本回答."""
    question = state["question"]
    analysis = state.get("analysis_result", "")
    docs = state.get("retrieved_docs", [])
    case_data = state.get("case_data", "")
    history_text = _format_history(state.get("messages", []))

    # 构建上下文
    parts = []
    if history_text:
        parts.append(history_text.rstrip())
    parts.append(f"[问题分析]\n{analysis}")

    if docs:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"[参考条文{i} 来源:{source}]\n{doc.page_content}")
        parts.append("[检索到的法律条文]\n" + "\n\n".join(context_parts))
    else:
        parts.append("[检索到的法律条文]\n无相关文档命中,请基于你的法律知识回答.")

    if case_data:
        parts.append(f"[案例库数据]\n{case_data}")

    human_msg = (
        f"用户咨询:\n{question}\n\n"
        + "\n\n".join(parts)
        + "\n\n再次强调:请直接分析并回答用户咨询,严禁罗列或复述条文原文."
        + (" 若有对话历史,请结合上下文理解本轮咨询的指代(如\"他\"\"那次\"等)." if history_text else "")
    )

    # 案例库查询类回答不需要"法律依据",否则模型会补法条,产生未落地引用
    system_prompt = _synthesis_prompt
    if case_data:
        system_prompt += (
            "\n\n**本次是案例库数据查询**:答案来自案例库,不是法律咨询."
            "法律依据段落只写\"本回答依据案例库查询结果,不引用具体法条.\","
            "不要引用任何法律条文或司法解释."
        )

    t0 = time.time()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg),
    ]
    # 流式生成:astream 使 astream_events 能捕获 on_chat_model_stream 事件
    full = ""
    async for chunk in llm.astream(messages):
        full += chunk.content
    print(f"[Synthesis] LLM 流式生成耗时: {time.time()-t0:.2f}s, 字数: {len(full)}")

    # 引用核验:把答案里引用的、但检索没召回的条文原文补进来
    docs = verify_and_attach(full, docs)

    return {"final_answer": full, "retrieved_docs": docs}
