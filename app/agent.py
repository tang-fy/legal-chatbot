import re
from typing import List, Optional

from . import config
from .tools import all_tools
from .output_parser import get_format_instructions, output_parser, LegalAnswerModel

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
)

#系统提示词
system_prompt = f"""你是一位资深的法律专家,精通中国2法律.你的任务是根据用户的问题或描述的
事件,提供专业,准确的法律分析和解答.

你可以使用一下工具:
1.search_legal_documents:在法律文档库(法律条文,司法解释,案例等)中检索相关内容.
2.query_case_database: 查询结构化的法律案例数据库.

**工作流程:**
- 如果用户提出了具体的法律问题(如"合同法地几条关于违约金?"),请先调用search_legal_documents
检索想关法条.
- 如果用户描述了一个事件或者情景(例如"我朋友借了我的钱不还,怎么办?"),你需要:
    1.分析事件涉及的法律关系(如合同,侵权,刑事等).
    2.调用search_legal_documents检索相关的法律依据和类似案例.
    3.结合检索结果和你的法律知识,判断该行为是否违法,并说明可能的法律后果.
- 如果用户询问具体案例,可调用query_case_database查询案例库.
- 如果问题无法通过工具解决,可以基于你的法律知识直接回答,但要注明"基于一般法律知识,未检索到具体条文".

**重要要求:**
- 回答必循客观,准确,引用法律条文时要注明出处(如<<中华人民共和国合同法>>第x条).
- 对于不确定的法律问题,应提示用户咨询专业律师.
- 请严格按照以下JSON格式输出(不要输出任何其他文本):
{get_format_instructions()}"""

agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt=system_prompt,
)

def ask(question: str, history: List[BaseMessage]) -> tuple[str, List[BaseMessage]]:
    """
    处理用户问题,返回最终回答和更新后的历史.
    """
    #新的API的输入是message列表:历史消息＋当前问题
    #history本身就是BaseMessage列表,可直接传入
    messages = list(history) + [HumanMessage(content=question)]

    #调用Agent
    result = agent.invoke({"messages": messages})

    #最终结果在消息列表最后一条
    output = result["messages"][-1].content

    #尝试解析JSON(容忍模型在JSON前后附加说明文字)
    parsed = _parse_legal_answer(output)
    if parsed is not None:
        answer = (
            f"法律分析:\n{parsed.analysis}\n\n"
            f"法律依据:\n{parsed.legal_basis}\n\n"
            f"结论:\n{parsed.conclusion}\n\n"
            f"免责声明:\n{parsed.disclaimer}\n\n"
        )
    else:
        answer = output

    #更新历史
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    history = history[-config.HISTORY_LIMIT:]
    return answer, history