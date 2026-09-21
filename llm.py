from langchain_Openai import ChatOpenAI
import os

def make_llm(model="inclusionai/ling-3.0-flash-vl:free", **kw):
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        **kw,
    )
llm = make_llm()


