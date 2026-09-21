from typing import Optional
from langchain_openai import ChatOpenAI
from config import MODEL, OPENROUTER_API_KEY


def make_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=4096,
        timeout=90,
        max_retries=3,
        **kwargs,
    )
