"""Tool definitions and registry for Aegis agents."""

import os
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List
import requests
from langchain_core.tools import StructuredTool

from config import MAX_OUTPUT_CHARS, TAVILY_API_KEY


def code_exec(code: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute Python code in an isolated subprocess and return stdout/stderr."""
    if not isinstance(code, str) or not code.strip():
        return {
            "success": False,
            "error_type": "INVALID_INPUT",
            "error": "Code parameter must be a non-empty string.",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }
    timeout = max(1, min(int(timeout), 30))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "script.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
            out, err = p.stdout or "", p.stderr or ""
            if len(out) > MAX_OUTPUT_CHARS:
                out = out[:MAX_OUTPUT_CHARS] + "\n... [Output truncated]"
            if len(err) > MAX_OUTPUT_CHARS:
                err = err[:MAX_OUTPUT_CHARS] + "\n... [Error output truncated]"
            return {
                "success": p.returncode == 0,
                "stdout": out,
                "stderr": err,
                "returncode": p.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error_type": "TIMEOUT",
                "error": f"Execution timed out after {timeout} seconds.",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "error_type": "EXECUTION_ERROR",
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }


def search(query: str) -> dict[str, Any]:
    """Search the web for queries using the Tavily search API."""
    try:
        from tavily import TavilyClient

        resp = TavilyClient(api_key=TAVILY_API_KEY).search(
            query=query, search_depth="basic", max_results=5
        )
        results = [
            {
                "title": i.get("title"),
                "url": i.get("url"),
                "content": i.get("content"),
                "score": i.get("score"),
            }
            for i in resp.get("results", [])
        ]
        return {"success": True, "query": query, "results": results}
    except Exception as e:
        return {"success": False, "error_type": "SEARCH_ERROR", "error": str(e)}


def calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """Perform basic arithmetic operations: add, subtract, multiply, divide."""
    ops = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b,
    }
    if operation not in ops:
        return {
            "success": False,
            "error_type": "INVALID_OPERATION",
            "error": f"operation must be one of {list(ops)}",
        }
    if operation == "divide" and b == 0:
        return {
            "success": False,
            "error_type": "DIVISION_BY_ZERO",
            "error": "division by zero",
        }
    return {"success": True, "expression": operation, "result": ops[operation]()}


def get_weather(city: str, unit: str = "celsius") -> dict[str, Any]:
    """Get the current weather forecast for a specified city."""
    if unit not in ("celsius", "fahrenheit"):
        return {"success": False, "error": "unit must be 'celsius' or 'fahrenheit'"}
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=15,
        ).json()
        if not geo.get("results"):
            return {"success": False, "error": f"city '{city}' not found"}
        loc = geo["results"][0]
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            timeout=15,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current_weather": True,
                "temperature_unit": unit,
            },
        ).json()["current_weather"]
        return {
            "success": True,
            "city": loc["name"],
            "temperature": w["temperature"],
            "unit": unit,
            "windspeed": w["windspeed"],
        }
    except Exception as e:
        return {"success": False, "error_type": "WEATHER_ERROR", "error": str(e)}


TOOL_REGISTRY: dict[str, Callable] = {
    "code_exec": code_exec,
    "search": search,
    "calculator": calculator,
    "get_weather": get_weather,
}


def _tool_doc(fn: Any) -> str:
    lines = (fn.__doc__ or "").strip().splitlines()
    return lines[0].strip() if lines else "No description available."


TOOL_DOCS: str = "\n".join(
    f"- {n}: {_tool_doc(f)}"
    for n, f in TOOL_REGISTRY.items()
)


def get_langchain_tools() -> List[StructuredTool]:
    """Return LangChain StructuredTool instances for all registered tools."""
    return [
        StructuredTool.from_function(fn, name=name, description=fn.__doc__ or name)
        for name, fn in TOOL_REGISTRY.items()
    ]
