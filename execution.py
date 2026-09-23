import json
import re
from typing import Any, Dict, List

from schemas import PlanStep
from tools import TOOL_REGISTRY

_PLACEHOLDER = re.compile(r"\{step_(\d+)_result(?:\.[a-zA-Z_0-9]+)?\}")


def primary_value(obs: dict[str, Any]) -> str:
    if "result" in obs:
        return str(obs["result"])
    if "temperature" in obs:
        return str(obs["temperature"])
    if "stdout" in obs:
        return obs["stdout"].strip()
    if "results" in obs:
        return "\n".join(f"{r['title']}: {r['content']}" for r in obs["results"])
    return json.dumps(obs)


def _resolve(value: Any, results: dict[int, str], numeric: bool) -> Any:
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value.strip())
        if whole and numeric:
            try:
                return float(results.get(int(whole.group(1)), ""))
            except ValueError:
                return value
        return _PLACEHOLDER.sub(
            lambda m: str(results.get(int(m.group(1)), m.group(0))), value
        )
    if isinstance(value, dict):
        return {k: _resolve(v, results, numeric) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, results, numeric) for v in value]
    return value


def run_step(step: PlanStep, results: dict[int, str]) -> dict[str, Any]:
    fn = TOOL_REGISTRY.get(step.tool)
    if fn is None:
        return {
            "success": False,
            "error_type": "UNKNOWN_TOOL",
            "error": f"no tool {step.tool!r}",
        }
    try:
        try:
            args = json.loads(step.args_json)
        except Exception:
            import ast
            args = ast.literal_eval(step.args_json)

        if not isinstance(args, dict):
            args = {}
        args = {
            k: _resolve(v, results, step.tool == "calculator") for k, v in args.items()
        }
        return fn(**args)
    except Exception as e:
        return {"success": False, "error_type": type(e).__name__, "error": str(e)}


def execute_plan(steps: List[PlanStep], results: dict[int, str], trace: list) -> None:
    for step in steps:
        obs = run_step(step, results)
        ok = obs.get("success", "error" not in obs)
        results[step.id] = primary_value(obs) if ok else f"ERROR: {obs.get('error')}"
        trace.append(
            {
                "step": step.id,
                "tool": step.tool,
                "args": step.args_json,
                "description": step.description,
                "observation": obs,
            }
        )
        if not ok:
            break


def synthesis_input(
    query: str, executed: List[PlanStep], results: dict[int, str], replans_left: int
) -> str:
    lines = [f"Task: {query}", "Executed steps:"]
    lines += [
        f"{s.id}. {s.tool}({s.args_json}) -> {results.get(s.id, '(not run)')}"
        for s in executed
    ]
    return "\n".join(lines + [f"Replans remaining: {replans_left}"])
