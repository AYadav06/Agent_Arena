import json
import time
from typing import Generator
import streamlit as st

import config
from agents import (
    run_plan_execute,
    run_react,
    stream_plan_execute,
    stream_react,
)
from llm import make_llm
from tools import TOOL_REGISTRY

st.set_page_config(
    page_title="Agent Arena",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .app-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .app-title-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #f8fafc;
        text-transform: uppercase;
    }
    .app-tag {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        background: #1e293b;
        color: #94a3b8;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .tools-group {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .tools-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        font-weight: 600;
        margin-right: 6px;
    }
    .tool-chip {
        background: #1e293b;
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }

    .command-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }

    .stMetric {
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }

    .output-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 18px 20px;
        line-height: 1.6;
        color: #f1f5f9;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tools_chips_html = "".join(
    f'<span class="tool-chip">{name}</span>'
    for name in TOOL_REGISTRY.keys()
)

st.markdown(
    f"""
    <div class="app-topbar">
        <div class="app-title-group">
            <div class="app-title">Agent Arena</div>
            <div class="app-tag">Evaluation Studio</div>
        </div>
        <div class="tools-group">
            <span class="tools-label">Registered Tools</span>
            {tools_chips_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Sidebar (Technical Configuration)
with st.sidebar:
    st.markdown("#### Engine Configuration")
    api_key_status = "Active (.env)" if config.OPENROUTER_API_KEY else "Not detected"
    st.caption(f"OpenRouter Credentials: **{api_key_status}**")

    api_key_override = st.text_input(
        "OpenRouter Key Override",
        type="password",
        value="",
        placeholder="sk-or-v1-...",
        help="Leave blank to use OPENROUTER_API_KEY from .env",
    )

    model_name = st.text_input(
        "OpenRouter Model Identifier",
        value=config.MODEL,
        help="E.g. google/gemini-2.0-flash-001, meta-llama/llama-3.3-70b-instruct",
    )

    tavily_status = "Active" if config.TAVILY_API_KEY else "Not detected"
    st.caption(f"Tavily Search: **{tavily_status}**")

    st.markdown("---")
    st.markdown("#### Execution Parameters")
    max_iterations = st.slider("Max ReAct Iterations", 2, 20, config.MAX_ITERATIONS)
    max_replans = st.slider("Max Plan-and-Execute Replans", 0, 3, config.MAX_REPLANS)


def text_stream_generator(text: str) -> Generator[str, None, None]:
    if not text:
        return
    words = text.split(" ")
    for idx, w in enumerate(words):
        yield w + (" " if idx < len(words) - 1 else "")
        time.sleep(0.015)


def display_metrics_bar(result: dict) -> None:
    usage = result.get("usage", {})
    trace = result.get("trace", [])
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Status", "Success" if result.get("success") else "Failed")
    with c2:
        st.metric("Latency", f"{usage.get('elapsed', 0):.2f}s")
    with c3:
        st.metric("LLM Invocations", str(usage.get("llm_calls", 0)))
    with c4:
        st.metric("Token Volume", f"{usage.get('total_tokens', 0):,}")
    with c5:
        st.metric("Tool Invocations", str(len(trace)))


def display_plan_steps(plan: list) -> None:
    if not plan:
        return
    st.markdown("##### Upfront Plan Formulated")
    for step in plan:
        step_id = step.get("id")
        tool = step.get("tool")
        desc = step.get("description")
        args = step.get("args_json", "{}")
        st.markdown(f"**Step {step_id}:** `{tool}` — *{desc}* (`{args}`)")


def display_trace_steps(trace: list) -> None:
    if not trace:
        return
    with st.expander(f"Execution Trace ({len(trace)} steps)", expanded=False):
        for i, step in enumerate(trace, 1):
            tool_name = step.get("tool") or step.get("tool_name", "unknown")
            thought = step.get("thought") or step.get("description", "")
            obs = step.get("observation")
            args = step.get("args") or step.get("tool_args", {})

            st.markdown(f"**Step {i}: `{tool_name}`**")
            if thought:
                st.caption(f"Reasoning: {thought}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            st.json(args)
            st.caption("Output:")
            if isinstance(obs, (dict, list)):
                st.json(obs)
            else:
                st.code(str(obs) if obs is not None else "(None)")
            st.markdown("---")


prompt_examples = {
    "(Custom prompt)": "",
    "Tokyo Weather and Temperature Doubled": "What is the weather in Tokyo right now and what is that temperature multiplied by 2?",
    "Paris Weather Plus 15 Degrees": "What is the weather in Paris and what is that temperature plus 15?",
    "Fibonacci Sequence via Python Execution": "Write and execute Python code using code_exec to compute the first 15 Fibonacci numbers.",
    "2024 Physics Nobel Prize Investigation": "Search for who won the 2024 Nobel Prize in Physics and what was the discovery?",
}

prompt_container = st.container()

with prompt_container:
    col_agent, col_presets = st.columns([3, 2])

    with col_agent:
        selected_agent = st.segmented_control(
            "Architecture",
            options=["ReAct", "Plan-and-Execute", "Compare Both"],
            default="ReAct",
            help="Select the reasoning and execution architecture.",
        )

    with col_presets:
        preset_choice = st.selectbox(
            "Task Templates",
            options=list(prompt_examples.keys()),
            index=0,
            help="Pre-configured multi-step benchmark prompts.",
        )

    default_text = prompt_examples.get(preset_choice, "")

    query = st.text_area(
        "Query Prompt",
        value=default_text,
        height=90,
        placeholder="Enter multi-step instruction or question...",
        key="query_input",
    )

    btn_col1, btn_col2 = st.columns([5, 1])
    with btn_col2:
        run_button = st.button("Run Workflow", type="primary", use_container_width=True)

if run_button:
    if not query.strip():
        st.warning("Please enter a query prompt before executing.")
        st.stop()

    effective_key = api_key_override.strip() or config.OPENROUTER_API_KEY
    if not effective_key:
        st.error(
            "Missing OpenRouter API Key. Configure OPENROUTER_API_KEY in .env or the sidebar."
        )
        st.stop()

    active_llm = make_llm(model=model_name.strip(), api_key=effective_key)

    if selected_agent == "ReAct":
        st.markdown("#### ReAct Agent")
        status_box = st.status("ReAct agent initializing...", expanded=True)
        live_trace_placeholder = status_box.empty()
        final_res = None
        last_step_count = 0

        for event in stream_react(query, llm=active_llm, max_iterations=max_iterations):
            if event["type"] == "update":
                trace = event.get("trace", [])
                if len(trace) > last_step_count:
                    last_step_count = len(trace)
                    latest = trace[-1]
                    status_box.update(
                        label=f"Step {last_step_count}: Running tool {latest.get('tool_name')}...",
                        state="running",
                    )
                with live_trace_placeholder.container():
                    for idx, s in enumerate(trace, 1):
                        st.markdown(f"**Step {idx}:** `{s.get('tool_name')}`")
                        if s.get("thought"):
                            st.caption(f"Reasoning: {s.get('thought')}")
                        if s.get("observation") is not None:
                            st.caption(f"Observation: {str(s.get('observation'))[:140]}...")

            elif event["type"] == "final":
                final_res = event["result"]
                err = event.get("error")
                if err:
                    status_box.update(label="Terminated with error", state="error")
                    st.error(err)
                else:
                    status_box.update(label="Execution completed", state="complete")

        if final_res:
            answer = final_res.get("answer", "")
            if answer:
                st.markdown("##### Output")
                st.write_stream(text_stream_generator(answer))
            elif not final_res.get("error"):
                st.warning("Workflow finished but returned an empty response.")

            st.markdown("---")
            display_metrics_bar(final_res)
            display_trace_steps(final_res.get("trace", []))

    elif selected_agent == "Plan-and-Execute":
        st.markdown("#### Plan-and-Execute Architecture")
        status_box = st.status("Formulating upfront plan...", expanded=True)
        plan_placeholder = status_box.empty()
        final_res = None

        for event in stream_plan_execute(query, llm=active_llm, allow_replan=(max_replans > 0)):
            if event["type"] == "phase":
                phase = event["phase"]
                if phase == "plan":
                    steps = event["data"].get("pending", [])
                    status_box.update(
                        label=f"Plan generated ({len(steps)} steps) -> Executing tools in Python...",
                        state="running",
                    )
                    with plan_placeholder.container():
                        st.markdown("**Formulated Plan:**")
                        for s in steps:
                            st.markdown(f"- Step {s.id}: `{s.tool}` — {s.description}")
                elif phase == "execute":
                    status_box.update(label="Tool execution finished -> Synthesizing output...", state="running")
                elif phase == "synthesize":
                    status_box.update(label="Synthesizing final response...", state="running")

            elif event["type"] == "final":
                final_res = event["result"]
                err = event.get("error")
                if err:
                    status_box.update(label="Terminated with error", state="error")
                    st.error(err)
                else:
                    status_box.update(label="Execution completed", state="complete")

        if final_res:
            answer = final_res.get("answer", "")
            if answer:
                st.markdown("##### Output")
                st.write_stream(text_stream_generator(answer))
            elif not final_res.get("error"):
                st.warning("Workflow finished but returned an empty response.")

            st.markdown("---")
            display_metrics_bar(final_res)
            if final_res.get("plan"):
                display_plan_steps(final_res["plan"])
            display_trace_steps(final_res.get("trace", []))

    elif selected_agent == "Compare Both":
        st.markdown("#### Head-to-Head Comparison")
        col_react, col_pe = st.columns(2)

        with col_react:
            st.markdown("##### ReAct Workflow")
            react_status = st.status("Running ReAct agent...", expanded=True)
            react_placeholder = react_status.empty()
            react_res = None
            last_rc = 0

            for event in stream_react(query, llm=active_llm, max_iterations=max_iterations):
                if event["type"] == "update":
                    trace = event.get("trace", [])
                    if len(trace) > last_rc:
                        last_rc = len(trace)
                        react_status.update(
                            label=f"Step {last_rc}: {trace[-1].get('tool_name')}",
                            state="running",
                        )
                    with react_placeholder.container():
                        for s in trace:
                            st.caption(f"• {s.get('tool_name')}: {str(s.get('observation'))[:80]}...")
                elif event["type"] == "final":
                    react_res = event["result"]
                    react_status.update(label="ReAct completed", state="complete")

            if react_res:
                ans = react_res.get("answer", "")
                st.markdown("**Output:**")
                if ans:
                    st.write_stream(text_stream_generator(ans))
                else:
                    st.caption("(Empty answer)")
                display_metrics_bar(react_res)
                display_trace_steps(react_res.get("trace", []))

        with col_pe:
            st.markdown("##### Plan-and-Execute Workflow")
            pe_status = st.status("Running Plan-and-Execute agent...", expanded=True)
            pe_placeholder = pe_status.empty()
            pe_res = None

            for event in stream_plan_execute(query, llm=active_llm, allow_replan=(max_replans > 0)):
                if event["type"] == "phase":
                    phase = event["phase"]
                    if phase == "plan":
                        steps = event["data"].get("pending", [])
                        pe_status.update(label=f"Plan formulated ({len(steps)} steps)", state="running")
                        with pe_placeholder.container():
                            for s in steps:
                                st.caption(f"• Step {s.id}: {s.tool} - {s.description}")
                    elif phase == "execute":
                        pe_status.update(label="Tools executed", state="running")
                    elif phase == "synthesize":
                        pe_status.update(label="Synthesizing response...", state="running")
                elif event["type"] == "final":
                    pe_res = event["result"]
                    pe_status.update(label="Plan-and-Execute completed", state="complete")

            if pe_res:
                ans = pe_res.get("answer", "")
                st.markdown("**Output:**")
                if ans:
                    st.write_stream(text_stream_generator(ans))
                else:
                    st.caption("(Empty answer)")
                display_metrics_bar(pe_res)
                if pe_res.get("plan"):
                    display_plan_steps(pe_res["plan"])
                display_trace_steps(pe_res.get("trace", []))

        if react_res and pe_res:
            st.markdown("---")
            st.markdown("##### Architecture Scorecard")
            ru = react_res.get("usage", {})
            pu = pe_res.get("usage", {})
            comp_table = {
                "Metric": [
                    "Status",
                    "Elapsed Time (Latency)",
                    "Total LLM Calls",
                    "Total Tokens Used",
                    "Prompt Tokens",
                    "Completion Tokens",
                    "Tool Steps Executed",
                ],
                "ReAct Agent": [
                    "Success" if react_res.get("success") else "Error",
                    f"{ru.get('elapsed', 0):.2f}s",
                    str(ru.get("llm_calls", 0)),
                    f"{ru.get('total_tokens', 0):,}",
                    f"{ru.get('prompt_tokens', 0):,}",
                    f"{ru.get('completion_tokens', 0):,}",
                    str(len(react_res.get("trace", []))),
                ],
                "Plan-and-Execute Agent": [
                    "Success" if pe_res.get("success") else "Error",
                    f"{pu.get('elapsed', 0):.2f}s",
                    str(pu.get("llm_calls", 0)),
                    f"{pu.get('total_tokens', 0):,}",
                    f"{pu.get('prompt_tokens', 0):,}",
                    f"{pu.get('completion_tokens', 0):,}",
                    str(len(pe_res.get("trace", []))),
                ],
            }
            st.table(comp_table)
