from typing import Optional
from langchain_openai import ChatOpenAI
from config import MODEL, OPENROUTER_API_KEY


def make_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    default_headers = {
        "HTTP-Referer": "https://github.com/AYadav06/Agent_Arena",
        "X-Title": "Agent Arena",
    }
    custom_headers = kwargs.pop("default_headers", {})
    headers = {**default_headers, **custom_headers}

    return ChatOpenAI(
        model=model or MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=4096,
        timeout=90,
        max_retries=3,
        default_headers=headers,
        **kwargs,
    )

