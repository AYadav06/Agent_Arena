"""Tool definitions and registry for Aegis agents."""

import os
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional
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


def calculator(
    operation: Optional[str] = None,
    a: Optional[Any] = None,
    b: Optional[Any] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Perform basic arithmetic operations: add, subtract, multiply, divide."""
    # Resolve aliases for operands
    if a is None:
        a = kwargs.get("number1", kwargs.get("num1", kwargs.get("x")))
    if b is None:
        b = kwargs.get("number2", kwargs.get("num2", kwargs.get("y")))

    # Resolve expression string like "15 + 25" if provided
    expr = kwargs.get("expression")
    if expr and (a is None or b is None or operation is None):
        import re
        expr_str = str(expr).strip()
        m = re.match(r"^([\d\.]+)\s*([\+\-\*\/])\s*([\d\.]+)$", expr_str)
        if m:
            a, sym, b = float(m.group(1)), m.group(2), float(m.group(3))
            sym_map = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
            operation = sym_map.get(sym, operation)

    if operation:
        operation = str(operation).lower().strip()
        op_map = {
            "+": "add",
            "plus": "add",
            "sum": "add",
            "-": "subtract",
            "minus": "subtract",
            "*": "multiply",
            "times": "multiply",
            "x": "multiply",
            "/": "divide",
        }
        operation = op_map.get(operation, operation)

    try:
        a_num = float(a) if a is not None else 0.0
        b_num = float(b) if b is not None else 0.0
    except (ValueError, TypeError):
        return {
            "success": False,
            "error_type": "INVALID_OPERANDS",
            "error": f"Operands 'a' ({a!r}) and 'b' ({b!r}) must be numbers.",
        }

    ops = {
        "add": lambda: a_num + b_num,
        "subtract": lambda: a_num - b_num,
        "multiply": lambda: a_num * b_num,
        "divide": lambda: a_num / b_num,
    }
    if operation not in ops:
        return {
            "success": False,
            "error_type": "INVALID_OPERATION",
            "error": f"operation must be one of {list(ops)}",
        }
    if operation == "divide" and b_num == 0:
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


TOOL_DOCS: str = (
    "- calculator(operation, a, b): Perform arithmetic. operation is 'add', 'subtract', 'multiply', or 'divide'; a and b are numbers.\n"
    "- code_exec(code): Run Python code in an isolated sandbox. code is a python script string.\n"
    "- search(query): Search the live web. query is a search string.\n"
    "- get_weather(city, unit='celsius'): Fetch current meteorological data for a city."
)


def get_langchain_tools() -> List[StructuredTool]:
    """Return LangChain StructuredTool instances for all registered tools."""
    return [
        StructuredTool.from_function(fn, name=name, description=fn.__doc__ or name)
        for name, fn in TOOL_REGISTRY.items()
    ]
