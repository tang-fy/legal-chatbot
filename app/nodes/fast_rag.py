"""Fast RAG 节点:单次 LLM 调用直连 RAG.

适用于普通法律咨询,跳过 Agent ReAct 循环.
逻辑迁自原 agent.py 的 _fast_rag_answer().
"""
import re
import time

from langchain_core.messages import SystemMessage, HumanMessage

from .. import config
from ..state import LegalState
from ..vector_store import similarity_search
from ..citations import verify_and_attach
from ..output_parser import get_format_instructions
from ..llm import llm
from .synthesis import _format_history  # 复用多轮历史拼接逻辑(单一来源)

# Fast Path 系统提示词(与原 agent.py 保持一致)
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


# 文档库只覆盖中国大陆法律,域外法问题直接拒答,避免用中国法条硬答
# 地区名单按常见程度排列,长名放在短名之前(如"印度尼西亚"先于"印度"),
# 避免"印度尼西亚"只命中"印度"后漏掉整词判断
_FOREIGN_RE = re.compile(
    r"(美国|加州|纽约|新加坡|日本|英国|德国|法国|韩国|朝鲜|欧盟|欧洲|北美|澳洲"
    r"|香港|澳门|台湾|澳大利亚|新西兰|加拿大|墨西哥|巴西|阿根廷|智利|秘鲁|哥伦比亚"
    r"|马来西亚|泰国|越南|印度尼西亚|印度|菲律宾|柬埔寨|老挝|缅甸|斯里兰卡|孟加拉"
    r"|巴基斯坦|俄罗斯|乌克兰|白俄罗斯|哈萨克斯坦|蒙古"
    r"|意大利|西班牙|葡萄牙|荷兰|比利时|瑞士|瑞典|挪威|丹麦|芬兰|波兰|希腊|奥地利"
    r"|爱尔兰|捷克|匈牙利|罗马尼亚"
    r"|沙特|阿联酋|迪拜|阿布扎比|卡塔尔|以色列|伊朗|土耳其|埃及|南非|尼日利亚|肯尼亚|摩洛哥"
    r"|域外法|境外法|外国法|哪国法律|外国法院|海外法律)"
)

# 问题虽然提到外国,但明确问的是中国法的效力/中国法院的态度,不能一拒了之
# 例:"中国公民在美国离婚,国内法院承认吗"应依据中国法(涉外民事诉讼程序)回答
_DOMESTIC_ANCHOR = re.compile(
    r"((中国|我国|国内)(法院|法律|境内)|涉外民事关系法律适用)"
)


def _is_foreign_question(question: str) -> bool:
    """是否属于文档库不覆盖的域外法问题."""
    if not _FOREIGN_RE.search(question):
        return False
    if _DOMESTIC_ANCHOR.search(question):
        print("[FastPath] 提及域外但落点是中国法,按国内法律回答")
        return False
    return True


_FOREIGN_REFUSAL = (
    "行为分析:\n您咨询的是中国大陆以外国家或地区的法律问题,"
    "本系统的法律文档库只覆盖中国大陆现行法律,未检索到可适用的域外法律文档,无法据此分析.\n\n"
    "法律后果:\n无法基于现有法律文档库判断该域外法问题.\n\n"
    "行动建议:\n建议您咨询当地执业律师,或查阅当地官方法律文本.\n\n"
    "法律依据:\n无.\n\n"
    "免责声明:\n本回答仅供参考,不构成法律建议.具体法律问题请咨询专业律师.\n\n"
)


def fast_rag_node(state: LegalState) -> dict:
    """Fast Path 节点:检索 + 1 次 LLM 调用."""
    question = state["question"]

    # 域外法不在文档库覆盖范围,直接拒答,不检索也不调 LLM
    if _is_foreign_question(question):
        print("[FastPath] 命中域外法,直接拒答")
        return {"final_answer": _FOREIGN_REFUSAL, "retrieved_docs": []}

    history_text = _format_history(state.get("messages", []))

    # 1. 检索文档(直接调用 vector_store,不走 @tool 包装)
    t0 = time.time()
    docs = similarity_search(
        question, k=config.RAG_TOP_K, threshold=config.RAG_SCORE_THRESHOLD
    )
    print(f"[Timing] FastPath RAG 检索: {time.time()-t0:.2f}s")

    if not docs:
        return {
            "final_answer": (
                "行为分析:\n未检索到与您问题相关的法律文档,无法基于现有文档库进行定性分析.\n\n"
                "法律后果:\n无法基于现有法律文档库判断.\n\n"
                "行动建议:\n建议您携带相关材料咨询专业律师获取专业法律意见.\n\n"
                "法律依据:\n无.\n\n"
                "免责声明:\n本回答仅供参考,不构成法律建议.具体法律问题请咨询专业律师.\n\n"
            ),
            "retrieved_docs": [],
        }

    # 2. 构建消息
    t1 = time.time()
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[参考条文{i} 来源:{source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    human_msg = (
        (history_text if history_text else "")
        + f"用户咨询:\n{question}\n\n"
        + f"[作为参考,以下是与本问题可能相关的法律条文,请只挑选直接相关的引用]\n{context}\n\n"
        + f"再次强调:以上条文仅供参考,请直接分析并回答用户咨询,严禁罗列或复述条文原文。"
        + (" 若有对话历史,请结合上下文理解本轮咨询的指代(如\"他\"\"那次\"等)。" if history_text else "")
    )
    messages = [
        SystemMessage(content=_rag_system_prompt),
        HumanMessage(content=human_msg),
    ]
    response = llm.invoke(messages)
    print(f"[Timing] FastPath LLM 生成: {time.time()-t1:.2f}s")

    # 引用核验:把答案里引用的、但检索没召回的条文原文补进来
    docs = verify_and_attach(response.content, docs)

    # 把原始输出写进 state,parse_node 负责解析
    return {
        "final_answer": response.content,
        "retrieved_docs": docs,
    }
