from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

#定义期望的结构化输出模型
class LegalAnswerModel(BaseModel):
    behavior_analysis: str = Field(description="对用户行为的法律分析:行为性质(民事/行政/刑事)、是否违法、构成要件")
    consequences: str = Field(description="法律后果:按情节轻重分情形说明可能承担的民事/行政/刑事责任")
    advice: str = Field(description="行动建议:如何避免违法或减轻法律责任,给出可执行的具体做法")
    legal_basis: str = Field(description="引用的1-3条核心法律条文:法律名+条款号+一句话概括")
    disclaimer: str = Field(
        default="本回答仅供参考,不构成正式法律意见.",
        description="免责声明"
    )

#创建Pydantic输出解析器,它会把LLM返回的JSON字符串解析为LegalAnswerModel对象
output_parser = PydanticOutputParser(pydantic_object=LegalAnswerModel)

def get_format_instructions() -> str:
    """
    返回格式说明字符串,用于拼接到Prompt中,告诉LLM必须按照指定的JSON格式输出.
    例如会包含字段名,类型和描述.
    """
    return output_parser.get_format_instructions()