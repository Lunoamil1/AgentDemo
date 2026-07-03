import random

from rag.rag_service import RagSummarizeService
from langchain_core.tools import tool
rag=RagSummarizeService()

@tool(description="从向量库中检索参考资料")
def rag_summarize(query:str):
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气情况,返回消息字符串")
def get_weather(city:str):
    return f"{city}天气晴朗,气温26摄氏度"


@tool(description="获取指定城市的名称,返回消息字符串")
def get_city():
    return random.choice(["北京","上海","广州","深圳","成都"])



