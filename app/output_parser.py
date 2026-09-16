from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

#定义期望的结构化输出模型
class LegalAnswerModel(BaseModel):
    analysis: str = Field(description="对用户问题或事件的详细法律分析")
    legal_basis: str = Field(description="引用的法律条文、司法解释或案例依据")
    conclusion: str = Field(description="最终结论,明确回答是否违法或法律后果")
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