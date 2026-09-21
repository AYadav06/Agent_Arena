
import os
from dotenv import load_dotenv

load_dotenv()

def _get_config(key: str, default: str = "") -> str:
    """Retrieve config from environment variables, falling back to Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            if key in st.secrets:
                return str(st.secrets[key])
    except Exception:
        pass
    return default


MODEL = _get_config("MODEL", "inclusionai/ling-3.0-flash-vl:free")
OPENROUTER_API_KEY = _get_config("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = _get_config("TAVILY_API_KEY", "")
MAX_ITERATIONS = int(_get_config("MAX_ITERATIONS", "8"))
MAX_REPLANS = 1
MAX_OUTPUT_CHARS = 8000
