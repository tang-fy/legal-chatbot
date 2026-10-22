"""SQL 节点:NL2SQL 查询 MySQL 案例库.

逻辑由原 tools.py 的案例查询逻辑迁入,直接作为 graph 节点调用,
底层仍复用 tools.py 的 get_db / _get_sql_llm / _extract_sql.
"""
import time

from ..state import LegalState
from ..tools import get_db, _get_sql_llm, _extract_sql


def sql_node(state: LegalState) -> dict:
    """NL2SQL 节点:把用户问题转 SQL,查询案例库."""
    question = state["question"]

    try:
        from langchain_classic.chains import create_sql_query_chain
        db = get_db()
        sql_llm = _get_sql_llm()
        chain = create_sql_query_chain(sql_llm, db)

        t0 = time.time()
        raw_output = chain.invoke({"question": question})
        sql_query = _extract_sql(raw_output)
        result = db.run(sql_query)
        print(f"[SQL] NL2SQL 耗时 {time.time()-t0:.2f}s,SQL: {sql_query[:80]}")

        return {"case_data": f"生成的SQL:\n{sql_query}\n\n查询结果:\n{result}"}
    except Exception as e:
        print(f"[SQL] 查询失败: {e}")
        return {"case_data": f"数据库查询失败: {e}"}
