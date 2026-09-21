import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import MAX_ITERATIONS, MAX_REPLANS
from execution import execute_plan, synthesis_input
from llm import make_llm
from prompts import PLANNER_SYSTEM, REACT_SYSTEM, SYNTH_SYSTEM
from schemas import Plan, PlanStep, Synthesis, Usage, make_result
from tools import get_langchain_tools

TOOLS = get_langchain_tools()


def _track(usage: Usage, msg: Any) -> None:
    um = getattr(msg, "usage_metadata", None) or {}
    usage.add(um.get("input_tokens", 0), um.get("output_tokens", 0))


def _trace_from(messages: list) -> List[dict[str, Any]]:
    trace: List[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                item = {
                    "iteration": len(trace) + 1,
                    "thought": m.content,
                    "tool_name": tc["name"],
                    "tool_args": tc["args"],
                    "observation": None,
                }
                by_id[tc["id"]] = item
                trace.append(item)
        elif m.type == "tool" and getattr(m, "tool_call_id", None) in by_id:
            by_id[m.tool_call_id]["observation"] = m.content
    return trace


def build_react_graph(llm=None, usage: Optional[Usage] = None):
    """Compile the ReAct LangGraph agent workflow."""
    active_llm = llm or make_llm()
    bound = active_llm.bind_tools(TOOLS)

    def agent(state: MessagesState):
        reply = bound.invoke([SystemMessage(REACT_SYSTEM)] + state["messages"])
        if usage is not None:
            _track(usage, reply)
        return {"messages": [reply]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def run_react(
    query: str, llm=None, max_iterations: int = MAX_ITERATIONS
) -> Dict[str, Any]:
    """Execute a query using the ReAct agent synchronously."""
    started, usage = time.time(), Usage()
    app = build_react_graph(llm or make_llm(), usage)
    state, error = {"messages": []}, None
    try:
        for state in app.stream(
            {"messages": [HumanMessage(query)]},
            {"recursion_limit": max_iterations * 2},
            stream_mode="values",
        ):
            pass
    except GraphRecursionError:
        error = f"Stopped: hit the {max_iterations}-iteration limit without a final answer."
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    msgs = state.get("messages", [])
    answer = (
        msgs[-1].content
        if msgs and msgs[-1].type == "ai" and not error and not getattr(msgs[-1], "tool_calls", None)
        else ""
    )
    return make_result("react", query, answer, _trace_from(msgs), usage, started, error=error)


def stream_react(query: str, llm=None, max_iterations: int = MAX_ITERATIONS):
    """Stream execution events and final result from the ReAct agent."""
    started, usage = time.time(), Usage()
    app = build_react_graph(llm or make_llm(), usage)
    state, error = {"messages": []}, None
    try:
        for state in app.stream(
            {"messages": [HumanMessage(query)]},
            {"recursion_limit": max_iterations * 2},
            stream_mode="values",
        ):
            msgs = state.get("messages", [])
            if msgs:
                yield {
                    "type": "update",
                    "trace": _trace_from(msgs),
                    "last_message": msgs[-1],
                    "usage": usage,
                }
    except GraphRecursionError:
        error = f"Stopped: hit the {max_iterations}-iteration limit without a final answer."
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    msgs = state.get("messages", [])
    answer = (
        msgs[-1].content
        if msgs and msgs[-1].type == "ai" and not error and not getattr(msgs[-1], "tool_calls", None)
        else ""
    )
    final_res = make_result("react", query, answer, _trace_from(msgs), usage, started, error=error)
    yield {
        "type": "final",
        "result": final_res,
        "answer": answer,
        "error": error,
    }

class PEState(TypedDict):
    query: str
    pending: List[PlanStep]
    executed: List[PlanStep]
    results: dict[int, str]
    trace: list
    answer: str
    replans: int


def build_plan_graph(llm=None, usage: Optional[Usage] = None, allow_replan: bool = True):
    active_llm = llm or make_llm()

    def structured(schema, system: str, prompt: str):
        runnable = active_llm.with_structured_output(
            schema, method="function_calling", include_raw=True
        )
        out = runnable.invoke([SystemMessage(system), HumanMessage(prompt)])
        if usage is not None:
            _track(usage, out["raw"])
        if out["parsed"] is None:
            raise ValueError(
                f"Model returned an invalid {schema.__name__}: {out.get('parsing_error')}"
            )
        return out["parsed"]

    def plan(state: PEState):
        return {
            "pending": structured(Plan, PLANNER_SYSTEM, state["query"]).steps
        }

    def execute(state: PEState):
        results, trace = dict(state["results"]), list(state["trace"])
        execute_plan(state["pending"], results, trace)
        return {
            "results": results,
            "trace": trace,
            "pending": [],
            "executed": state["executed"] + state["pending"],
        }

    def synthesize(state: PEState):
        left = (MAX_REPLANS - state["replans"]) if allow_replan else 0
        syn = structured(
            Synthesis,
            SYNTH_SYSTEM,
            synthesis_input(state["query"], state["executed"], state["results"], left),
        )
        if not syn.done and syn.extra_steps and left > 0:
            return {"pending": syn.extra_steps, "replans": state["replans"] + 1}
        return {"answer": syn.answer, "pending": []}

    g = StateGraph(PEState)
    g.add_node("plan", plan)
    g.add_node("execute", execute)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "synthesize")
    g.add_conditional_edges(
        "synthesize",
        lambda s: "execute" if s["pending"] else END,
        {"execute": "execute", END: END},
    )
    return g.compile()


def run_plan_execute(query: str, llm=None, allow_replan: bool = True) -> Dict[str, Any]:
    started, usage = time.time(), Usage()
    app = build_plan_graph(llm or make_llm(), usage, allow_replan)
    init: PEState = {
        "query": query,
        "pending": [],
        "executed": [],
        "results": {},
        "trace": [],
        "answer": "",
        "replans": 0,
    }
    try:
        final = app.invoke(init, {"recursion_limit": 3 * (MAX_REPLANS + 2) + 2})
    except Exception as e:
        return make_result(
            "plan_execute",
            query,
            "",
            [],
            usage,
            started,
            error=f"{type(e).__name__}: {e}",
        )
    return make_result(
        "plan_execute",
        query,
        final.get("answer", ""),
        final.get("trace", []),
        usage,
        started,
        plan=[s.model_dump() for s in final.get("executed", [])],
    )


def stream_plan_execute(query: str, llm=None, allow_replan: bool = True):
    started, usage = time.time(), Usage()
    app = build_plan_graph(llm or make_llm(), usage, allow_replan)
    init: PEState = {
        "query": query,
        "pending": [],
        "executed": [],
        "results": {},
        "trace": [],
        "answer": "",
        "replans": 0,
    }
    final_state = dict(init)
    try:
        for update in app.stream(
            init,
            {"recursion_limit": 3 * (MAX_REPLANS + 2) + 2},
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                final_state.update(node_output)
                yield {
                    "type": "phase",
                    "phase": node_name,
                    "data": node_output,
                    "state": final_state,
                    "usage": usage,
                }
    except Exception as e:
        final_res = make_result(
            "plan_execute",
            query,
            "",
            [],
            usage,
            started,
            error=f"{type(e).__name__}: {e}",
        )
        yield {
            "type": "final",
            "result": final_res,
            "answer": "",
            "error": f"{type(e).__name__}: {e}",
        }
        return

    final_res = make_result(
        "plan_execute",
        query,
        final_state.get("answer", ""),
        final_state.get("trace", []),
        usage,
        started,
        plan=[s.model_dump() for s in final_state.get("executed", [])],
    )
    yield {
        "type": "final",
        "result": final_res,
        "answer": final_state.get("answer", ""),
        "error": None,
    }
