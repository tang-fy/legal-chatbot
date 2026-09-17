import re
import time
from typing import List, Optional

from . import config
from .tools import all_tools
from .vector_store import similarity_search
from .output_parser import get_format_instructions, output_parser, LegalAnswerModel

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama


def _parse_legal_answer(output: str) -> Optional[LegalAnswerModel]:
    """
    解析模型输出.本地7B模型有时不会严格只输出JSON,
    会在JSON前后附加自然语言说明,因此先整体解析,
    失败再尝试提取文本中的第一个JSON对象解析.
    """
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

#初始化大语言模型
llm = ChatOllama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0.1, #较低的温度保证法律分析的严谨性
    num_ctx=8192,    #上下文窗口:4096装不下系统提示词+检索片段,会静默截断开头的系统指令导致模型罗列法条
    num_predict=1024, #限制最大生成token数,避免模型输出过长拖慢速度
)

# ========== Fast Path: 直接 RAG 回答 ==========
# 大部分法律问题只需要 RAG 检索 + LLM 生成回答,
# 跳过 Agent 的 ReAct 工具选择轮次,直接用 1 次 LLM 调用完成.

# 用于 Fast Path 的简化系统提示词(不含工具调用说明)
_rag_system_prompt = f"""你是一位经验丰富、善于向普通人解释法律问题的律师顾问。你的目标不是给用户罗列法条,而是像真人律师那样,针对用户描述的具体行为或情境,告诉TA:行为性质、是否违法、不同做法的不同后果、应如何合法应对。

**回答要求(必须严格遵守):**
1. 开头先用一句话定性用户行为(如"您描述的'开车撞死行人'涉嫌交通肇事罪"),不要先介绍或复述检索到的条文。
2. **行为分析**:结合用户描述的具体行为逐一分析法律性质和构成要件,明确"是否违法"。
3. **法律后果(分情形)**:按情节轻重分别说明后果(如未逃逸/逃逸/逃逸致人死亡的不同刑期)。
4. **行动建议**:给出可执行的合法做法(如事故后立即停车、报警、抢救伤者,切勿逃逸)。
5. 法律依据只引用 1-3 条**最核心**条文(法律名+条款号+一句话概括)。如果检索到的条文与案情不相关(如问撞人刑责却检索到信号灯、罚款条款),不要罗列它们,基于你自己的法律知识回答,并注明"相关条文未在检索结果中命中,依据一般法律知识"。
6. 全程说人话,严禁整段抄录或罗列法条原文。

**示例(请模仿这种风格回答):**

用户问:我在马路上开车撞死了人会被判几年?

期望回答:
{{"behavior_analysis":"您描述的'开车撞死行人'涉嫌交通肇事罪(《刑法》第133条)。是否构成犯罪,关键看事故责任划分与伤亡后果:负主要或全部责任且造成1人死亡,即构成交通肇事罪;若负同等或次要责任,则属一般交通事故,不构成犯罪,仅承担民事赔偿和行政责任。","consequences":"1. 不构成犯罪(同等/次要责任):仅承担民事赔偿和行政处罚(吊销驾驶证)。\\n2. 构成犯罪未逃逸:3年以下有期徒刑或拘役。\\n3. 肇事后逃逸或有其他特别恶劣情节:3-7年有期徒刑。\\n4. 因逃逸致人死亡:7年以上有期徒刑。\\n5. 民事上还需赔偿死亡赔偿金、丧葬费、被扶养人生活费等。","advice":"1. 事故后立即停车,保护现场,拨打120抢救伤者、122报警。\\n2. 千万不要逃逸,逃逸会使刑期升格为3-7年甚至7年以上。\\n3. 积极赔偿死者家属、争取谅解书,量刑时可从轻处罚。\\n4. 尽快委托刑事辩护律师介入。","legal_basis":"《刑法》第133条:交通肇事罪,违反交通运输管理法规致人死亡处3年以下有期徒刑,逃逸或情节恶劣处3-7年,因逃逸致人死亡处7年以上;《道路交通安全法》第101条:构成犯罪的依法追究刑事责任并吊销驾驶证。","disclaimer":"本回答仅供参考,不构成法律意见,涉嫌刑事犯罪请尽快委托专业律师。"}}

**再次强调:你的任务是分析用户的行为并给出后果和建议,绝对不要把检索到的条文原文罗列出来!**

请严格按照以下JSON格式输出(不要输出任何其他文本):
{get_format_instructions()}"""

# 判断问题是否可能需要 NL2SQL(查询结构化案例库)
_NL2SQL_KEYWORDS = re.compile(
    r"(案例库|判决书|裁决书|案件统计|纠纷类型统计|多少起|多少件|几起案件|几件案件|起案件|件案件)",
    re.IGNORECASE,
)

def _needs_sql(query: str) -> bool:
    """简单关键词判断问题是否需要 NL2SQL."""
    return bool(_NL2SQL_KEYWORDS.search(query))


def _fast_rag_answer(question: str) -> str:
    """
    Fast Path: 直接检索 + LLM 生成回答,跳过 Agent.
    只有 1 次 LLM 调用,比 Agent 模式快一倍以上.

    注意: 不使用 ChatPromptTemplate,因为 _rag_system_prompt 是
    已经格式化好的 f-string,内含 JSON schema 的花括号 {},
    ChatPromptTemplate 会把它当成模板变量解析导致报错.
    直接构建 SystemMessage + HumanMessage 即可.
    """
    # 1. 检索文档
    # 注意:必须直接调用 vector_store.similarity_search(返回 List[Document]),
    # 不能用 tools 里 @tool 包装的 search_legal_documents.invoke(),
    # 后者返回的是拼接好的字符串,无法访问 doc.metadata/doc.page_content
    t0 = time.time()
    docs = similarity_search(question, k=config.RAG_TOP_K)
    print(f"[Timing] FastPath RAG 检索: {time.time()-t0:.2f}s")

    # 2. 如果没有检索到相关文档,直接告知用户
    if not docs:
        return (
            "行为分析:\n未检索到与您问题相关的法律文档,无法基于现有文档库进行定性分析.\n\n"
            "法律后果:\n无法基于现有法律文档库判断.\n\n"
            "行动建议:\n建议您携带相关材料咨询专业律师获取专业法律意见.\n\n"
            "法律依据:\n无.\n\n"
            "免责声明:\n本回答仅供参考,不构成法律建议.具体法律问题请咨询专业律师.\n\n"
        )

    # 3. 构建消息(不用 ChatPromptTemplate,避免花括号冲突)
    t1 = time.time()
    # 给检索结果加上来源标记,便于 LLM 引用溯源
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[参考条文{i} 来源:{source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # 关键:措辞改为"用户咨询"+问题在前,参考条文在后,避免模型先介绍文档
    human_msg = (
        f"用户咨询:\n{question}\n\n"
        f"[作为参考,以下是与本问题可能相关的法律条文,请只挑选直接相关的引用]\n{context}\n\n"
        f"再次强调:以上条文仅供参考,请直接分析并回答用户咨询,严禁罗列或复述条文原文。"
    )
    messages = [
        SystemMessage(content=_rag_system_prompt),
        HumanMessage(content=human_msg),
    ]
    response = llm.invoke(messages)
    print(f"[Timing] FastPath LLM 生成: {time.time()-t1:.2f}s")

    # 4. 解析 JSON
    parsed = _parse_legal_answer(response.content)
    if parsed is not None:
        return (
            f"行为分析:\n{parsed.behavior_analysis}\n\n"
            f"法律后果:\n{parsed.consequences}\n\n"
            "行动建议:\n" + parsed.advice + "\n\n"
            f"法律依据:\n{parsed.legal_basis}\n\n"
            f"免责声明:\n{parsed.disclaimer}\n\n"
        )
    print(f"[Warn] FastPath LLM输出未能解析为JSON,返回原始输出前200字: {response.content[:200]}")
    return response.content


# ========== Agent 模式: 多工具 ReAct ==========

#系统提示词(Agent 模式用)
system_prompt = f"""你是一位经验丰富、善于向普通人解释法律问题的律师顾问。你的目标不是给用户罗列法条,而是像真人律师那样,针对用户描述的具体行为或情境,告诉TA:行为性质、是否违法、不同做法的不同后果、应如何合法应对。

你可以使用以下工具:
1.search_legal_documents:在法律文档库(法律条文,司法解释,案例)中检索相关内容.
2.query_case_database:查询结构化的法律案例数据库.

**工作流程:**
- 用户提出具体法律问题 → 先调用search_legal_documents检索相关法条.
- 用户描述事件或情境 → ①分析涉及的法律关系(民事/行政/刑事);②调用search_legal_documents检索法律依据;③结合检索结果和你的法律知识,判断各当事人行为的合法性及法律后果.
- 用户询问具体案例或案件数据 → 调用query_case_database查询案例库.
- 若检索到的条文与用户问题不相关(如问撞人刑责却检索到信号灯、罚款条款),不要照抄,基于自己的法律知识回答,并注明"相关条文未在检索结果中命中,依据一般法律知识".

**回答风格(非常重要):**
1. 开头先用一句话定性用户行为(如"您描述的'开车撞死行人'涉嫌交通肇事罪"),不要先介绍检索到的文档.
2. 结合案情中的具体当事人和具体行为逐一分析,说人话,严禁整段抄录或罗列法条原文.
3. **法律后果必须按情节轻重分情形说明**(如未逃逸/逃逸/逃逸致人死亡的不同刑期).
4. **必须给出可执行的合法做法**(如停车、报警、抢救、不要逃逸),让用户看完就知道怎么做.
5. 法律依据只引用 1-3 条最核心条文:法律名称+条款号+一句话概括.
6. 引用法律时注意:原《合同法》已废止,合同相关问题应引用《民法典》合同编.
7. 字数:行为分析≤400字、法律后果≤200字、行动建议≤200字、法律依据≤150字.
8. 对于不确定的法律问题,提示用户咨询专业律师.

请严格按照以下JSON格式输出(不要输出任何其他文本):
{get_format_instructions()}"""

agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt=system_prompt,
)


def _agent_answer(question: str, history: List[BaseMessage]) -> tuple[str, List[BaseMessage]]:
    """
    Agent 模式: 多工具 ReAct, 可能涉及多次 LLM 调用.
    适用于需要 NL2SQL 或复杂推理的问题.
    """
    messages = list(history) + [HumanMessage(content=question)]

    t0 = time.time()
    result = agent.invoke({"messages": messages})
    agent_elapsed = time.time() - t0
    print(f"[Timing] Agent 模式总耗时: {agent_elapsed:.1f}s")

    output = result["messages"][-1].content

    parsed = _parse_legal_answer(output)
    if parsed is not None:
        answer = (
            f"行为分析:\n{parsed.behavior_analysis}\n\n"
            f"法律后果:\n{parsed.consequences}\n\n"
            "行动建议:\n" + parsed.advice + "\n\n"
            f"法律依据:\n{parsed.legal_basis}\n\n"
            f"免责声明:\n{parsed.disclaimer}\n\n"
        )
    else:
        print(f"[Warn] Agent输出未能解析为JSON,返回原始输出前200字: {output[:200]}")
        answer = output

    #更新历史
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    history = history[-config.HISTORY_LIMIT:]
    return answer, history


def ask(question: str, history: List[BaseMessage]) -> tuple[str, List[BaseMessage]]:
    """
    处理用户问题,返回最终回答和更新后的历史.

    路由策略:
    - 如果问题可能需要 NL2SQL(关键词匹配) → 走 Agent 模式
    - 否则 → 走 Fast Path(直接 RAG + LLM,只需 1 次 LLM 调用)
    """
    t0 = time.time()

    # 路由判断
    if _needs_sql(question):
        print(f"[Router] 检测到可能需要 NL2SQL,走 Agent 模式")
        answer, history = _agent_answer(question, history)
    else:
        print(f"[Router] 普通法律问题,走 Fast Path")
        # Fast Path 也需要更新 history
        answer = _fast_rag_answer(question)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        history = history[-config.HISTORY_LIMIT:]

    total_elapsed = time.time() - t0
    print(f"[Timing] === 总耗时: {total_elapsed:.1f}s ===")
    return answer, history
