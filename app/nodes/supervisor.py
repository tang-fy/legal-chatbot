"""Supervisor 节点:路由决策.

判断问题是在问"案例库里的具体案件/统计"还是普通法律咨询,
写入 state["needs_sql"],后续由 graph 条件边决定走 Agent 还是 Fast Path.

路由规则(纯规则,零 LLM 调用):
1. 显式数据意图词:案例库/案件统计/多少起/判决书...
2. 具体案件名:"张三诉李四...案"
3. "案/案件" + 强数据线索:判决结果/判了多久/几起/几件...

第 3 条刻意不放宽到"案 + 法院/多少/多久",否则
"离婚案件法院一般怎么判""刑事案件请律师多少钱"会被误判为查库.
多轮指代(如"那第二个呢")仍未覆盖,需要时再加 LLM 兜底路由.
"""
import re

from ..state import LegalState

# 显式指向案例库/结构化的说法
_EXPLICIT_DB = re.compile(
    r"(案例库|案例数据|案件统计|纠纷类型统计|判决书|裁决书"
    r"|多少起|多少件|几起案件|几件案件|起案件|件案件)",
    re.IGNORECASE,
)

# 具体案件名:"张三诉李四合同纠纷案"
_CASE_NAME = re.compile(r"\S{2,}诉\S{2,}.*?案")

# "王五盗窃案是哪个法院审理的":案件名+法院问法
# 不匹配"离婚案件去哪个法院起诉"(中间多了"去",属于程序咨询)
_COURT_ON_CASE = re.compile(r"案(?:件)?(?:是|由|在)?(?:哪个|哪家|什么)法院")

# 具体案件名:人名/公司名 + 案由 + 案,如"王五盗窃案""张三的离婚案"
# 单独写一条是为了配合"审理法院/承办法官/在哪审理"这类不含"哪个法院"的问法
_SPECIFIC_CASE = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9]{2,4})"
    r"(?:盗窃|诈骗|抢劫|抢夺|故意伤害|交通肇事|危险驾驶|合同纠纷|民间借贷|劳动争议"
    r"|走私|贩毒|受贿|贪污|故意杀人|敲诈勒索|寻衅滋事|职务侵占|集资诈骗"
    r"|非法吸收公众存款|离婚|侵权|纠纷)"
    r"案"
)
# 这些前缀是案由而非当事人,如"劳动争议案""合同纠纷案",不能当作具体案件
_GENERIC_CASE_PREFIX = {"劳动", "交通", "合同", "借款", "买卖", "租赁", "物业",
                        "医疗", "保险", "民间", "离婚", "侵权"}

# 不含"哪个法院"的个案问法:审理法院/承办法官/在哪审理
_CASE_COURT_INFO = re.compile(r"(审理法院|审判法院|承办法官|主审法官|在哪里审理|在哪审理)")

# 泛案由+法院问法(如"离婚案件由哪个法院管辖")属于程序咨询,不应查案例库
_GENERIC_COURT_QUERY = re.compile(
    r"(离婚|劳动|合同|借款|买卖|租赁|物业|医疗|保险|继承|侵权"
    r"|交通事故|劳动争议|合同纠纷)案(?:件)?(?:是|由|在)?(?:哪个|哪家|什么)法院"
)


def _has_specific_case(question: str) -> bool:
    """是否指名道姓地问某个具体案件(而非泛泛的案由咨询)."""
    return any(
        m.group(1) not in _GENERIC_CASE_PREFIX
        for m in _SPECIFIC_CASE.finditer(question)
    )

# 只有这些强线索才说明是在问"库里的某个案件",而不是泛泛的法律咨询
_STRONG_CASE_CUE = re.compile(
    r"(判决结果|判了多久|判了几年|判了多少|判处"
    r"|几起|几件|涉及哪些法院|哪些法院)",
    re.IGNORECASE,
)


def _match_reason(question: str) -> str:
    """命中哪条规则,用于日志与误判排查."""
    if _EXPLICIT_DB.search(question):
        return "显式数据意图词"
    if _CASE_NAME.search(question):
        return "具体案件名"
    if _COURT_ON_CASE.search(question) and _has_specific_case(question):
        return "具体案件+法院问法"
    if _COURT_ON_CASE.search(question) and not _GENERIC_COURT_QUERY.search(question):
        return "案件指称+法院问法"
    if _has_specific_case(question) and _CASE_COURT_INFO.search(question):
        return "具体案件+审理信息"
    if "案" in question and _STRONG_CASE_CUE.search(question):
        return "案件指称+强数据线索"
    return ""


def supervisor_node(state: LegalState) -> dict:
    """路由决策节点:判断是否需要走 NL2SQL 查询案例库."""
    question = state["question"]
    reason = _match_reason(question)
    needs_sql = bool(reason)

    if needs_sql:
        print(f"[Router] 走 Agent 模式(命中: {reason})")
    else:
        print(f"[Router] 普通法律问题,走 Fast Path")

    return {"needs_sql": needs_sql}
