
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "inclusionai/ling-3.0-flash-vl:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))
MAX_REPLANS = 1
MAX_OUTPUT_CHARS = 8000
