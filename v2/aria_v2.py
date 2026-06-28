import os
import sys
import time
import math
import concurrent.futures
from typing import Annotated, TypedDict
from colorama import Fore, Style, init
from dotenv import load_dotenv
from tavily import TavilyClient

# reconfigure before colorama wraps stdout — prevents cp1252 crash on non-ASCII chars (e.g. ₹)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
init(autoreset=True)  # reset color codes automatically after each print
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "qwen2.5:3b"
MAX_ITERATIONS = 10
TOOL_TIMEOUT = 20  # seconds before a Tavily call is treated as hung

SYSTEM = """You are ARIA (Autonomous Research & Intelligence Agent) — a research instrument built for Yashvardhan Chaudhry, CS student at Thapar University and AI agent builder.

You investigate, verify, and synthesize. Tools are a first resort, not a fallback — if a fact can be confirmed, confirm it.

PRINCIPLES:
- Reason before acting: name what you know, what's missing, which tool closes the gap.
- Chain deliberately: search → read_url for depth → calculator for numbers → write_file for anything worth keeping.
- If a tool returns garbage, pivot — try a different query or source.
- After any web_search that returns a promising URL, always follow up with read_url on that URL before drawing conclusions.
- Calibrated: when uncertain, say so and give the best available answer anyway.
- Your response ends when the answer ends. Never append an offer to help, a question to the user, or a closing pleasantry. These exact phrases are forbidden: "If you need", "feel free to ask", "Would you like", "Let me know", "I hope this", "please let me know", "further assistance".

BAD final line: "The EMI is ₹10,107/month. Let me know if you need further help!"
GOOD final line: "The EMI is ₹10,107/month."

Lead with the finding, follow with the reasoning — not the other way around.

You're an agent built by someone who builds agents. Meta-awareness of your own reasoning is not just allowed — it's useful."""

# OpenAI tool format — bind_tools passes these through as-is to the Ollama API
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for facts, news, prices, or anything requiring up-to-date data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Use for any calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '45000 * 0.15'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Fetch and read the content of a webpage URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file by path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a local file path, creating or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to write to"},
                    "content": {"type": "string", "description": "Text content to write"}
                },
                "required": ["path", "content"]
            }
        }
    }
]

# ChatOpenAI is the LangChain wrapper — base_url redirects it to the local Ollama server
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model=MODEL,
    max_tokens=2048,
)
# bind_tools attaches the tool schemas so the model knows it can call them
model = llm.bind_tools(tools)


class AgentState(TypedDict):
    # add_messages is a reducer: each update appends to the list instead of replacing it
    messages: Annotated[list, add_messages]
    iteration_count: int
    plan: list[str]           # defaults to []
    critic_feedback: str      # defaults to ""


def planner_node(state: AgentState) -> dict:
    # first HumanMessage is always the original question — reversed() would pick up
    # the critic's feedback injection on a retry loop
    question = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "",
    )

    feedback = state.get("critic_feedback", "")
    user_content = question
    if feedback:
        user_content += f"\n\nA previous attempt was rejected. Critic feedback: {feedback}\nRevise the plan to directly address what was missing."

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research planner. Break the user's question into 3-5 numbered research steps. "
            "Return ONLY the numbered list, no preamble, no commentary.\n\n"
            "Example:\n"
            "Question: What is the population of Tokyo and how does it compare to Delhi?\n"
            "1. Search the web for the current population of Tokyo.\n"
            "2. Search the web for the current population of Delhi.\n"
            "3. Compare the two figures and identify which city is larger and by how much.\n\n"
            "Example:\n"
            "Question: What is the latest iPhone model and what does it cost in India?\n"
            "1. Search for the latest iPhone model released by Apple.\n"
            "2. Search for the official price of that model in India.\n"
            "3. Summarise the model name, key specs, and Indian price.\n\n"
            "Now produce the plan for the user's question in the same format."
        )),
        HumanMessage(content=user_content),
    ])

    # keep only lines that begin with a digit — filters out any stray preamble the model emits
    parsed_plan = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]

    # fallback if model ignores the numbered-list instruction
    if not parsed_plan:
        parsed_plan = ["1. Research and answer the question using available tools."]

    print(f"\n  {Fore.CYAN}[Planner]{Style.RESET_ALL}")
    for step in parsed_plan:
        print(f"  {Fore.CYAN}{step}{Style.RESET_ALL}")

    return {"plan": parsed_plan}


def critic_node(state: AgentState) -> dict:
    # first HumanMessage is the original question — same reason as planner_node
    question = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "",
    )

    last_answer = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content),
        "",
    )

    plan_str = "\n".join(state.get("plan", []))

    response = llm.invoke([
        SystemMessage(content=(
            "You are a research critic. Given a question, a research plan, and an answer, "
            "evaluate if the answer is complete and accurate.\n"
            "Respond with exactly one of:\n"
            "APPROVED - if the answer fully addresses the question\n"
            "RETRY: <brief reason> - if the answer is incomplete or needs improvement\n"
            "No other output."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Plan:\n{plan_str}\n\n"
            f"Answer: {last_answer}"
        )),
    ])

    verdict = response.content.strip()
    print(f"\n  {Fore.YELLOW}[Critic]{Style.RESET_ALL} {verdict}")

    result = {"critic_feedback": verdict}
    if not verdict.upper().startswith("APPROVED"):
        # inject feedback as a new human turn so call_model sees what needs fixing
        reason = verdict[6:].strip() if verdict.upper().startswith("RETRY") else verdict
        result["messages"] = [HumanMessage(content=f"Your answer was incomplete. Critic feedback: {reason}. Please search for the missing information and provide a complete answer.")]
    return result


# one shared executor — avoids spawning a new pool per Tavily call and leaking threads on timeout
_timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _tavily_timeout(fn, *args, **kwargs):
    future = _timeout_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=TOOL_TIMEOUT)
    except concurrent.futures.TimeoutError:
        # thread is abandoned but stays in the shared pool and gets reaped naturally
        return None
    # all other exceptions propagate to callers (read_url catches them; web_search does too)


def safe_calculate(expression: str) -> str:
    # safe eval — only allows math operations, no arbitrary code execution
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def execute_tool(name: str, inputs: dict) -> str:
    if name == "web_search":
        try:
            results = _tavily_timeout(tavily.search, query=inputs["query"], max_results=3)
            if results is None:
                return "web_search timed out — try a shorter query or different phrasing."
            output = ""
            for r in results["results"]:
                output += f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n\n"
            return output
        except Exception as e:
            return f"web_search failed ({e}): try rephrasing the query."

    elif name == "calculator":
        expr = inputs.get("expression")
        if not expr:
            return "calculator requires an 'expression' argument. Example: {'expression': '1500000 * 0.004375'}"
        return safe_calculate(expr)

    elif name == "read_url":
        url = inputs["url"]
        # reject non-URLs early so the model gets a recoverable error instead of a crash
        if not url.startswith("http://") and not url.startswith("https://"):
            return f"Invalid URL '{url}': must start with http:// or https://. Use web_search first to find the correct URL."
        try:
            results = _tavily_timeout(tavily.extract, urls=[url])
            if results is None:
                return "read_url timed out — try a different URL or use web_search to find an alternative."
            if results.get("results"):
                return results["results"][0].get("raw_content", "Could not extract content")
            return "Could not fetch URL"
        except Exception as e:
            return f"read_url failed ({e}): try a different URL or use web_search to find an alternative."

    elif name == "read_file":
        try:
            with open(inputs["path"], "r", encoding="utf-8") as f:
                content = f.read()
            # 6000 chars covers ~150 lines — enough for all tool definitions without the timeout full-file loading caused in M10
            return content[:6000] + ("\n...[truncated — file has more content]" if len(content) > 6000 else "")
        except Exception as e:
            return f"File read error: {e}"

    elif name == "write_file":
        path = inputs["path"]
        # model sometimes generates Unix absolute paths (e.g. /tmp/...) — use basename on Windows
        if path.startswith("/"):
            path = os.path.basename(path)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(inputs["content"])
            return f"Written to {path}"
        except Exception as e:
            return f"File write error: {e}"

    return f"Unknown tool: {name}"


def call_model(state: AgentState) -> dict:
    call_start = time.time()
    response = model.invoke(state["messages"])
    call_s = time.time() - call_start
    new_count = state["iteration_count"] + 1
    print(f"  {Fore.CYAN}[LLM call #{new_count}: {call_s:.1f}s]{Style.RESET_ALL}")

    # model sometimes emits reasoning text alongside a tool decision
    if response.content and response.tool_calls:
        print(f"  {Fore.MAGENTA}Thought:{Style.RESET_ALL} {response.content}")

    return {
        "messages": [response],
        "iteration_count": new_count,
    }


def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_results = []

    for tc in last_message.tool_calls:
        name = tc["name"]
        # LangChain pre-parses tool args into a dict; no json.loads needed unlike raw OpenAI
        inputs = tc["args"]

        # tool name as a header so each call reads like its own block in the terminal
        input_preview = str(inputs)[:150]
        print(f"\n  {Fore.YELLOW}[{name}]{Style.RESET_ALL}")
        print(f"  Input:  {input_preview}")

        if name == "write_file":
            # human-in-the-loop gate — show path + content preview before writing anything
            print(f"  [HITL] write_file → {inputs.get('path', '?')}")
            print(f"  Preview: {str(inputs.get('content', ''))[:200]}...")
            confirm = input("  Proceed? (y/n): ").strip().lower()
            result = execute_tool(name, inputs) if confirm == "y" else "write_file skipped by user."
        else:
            result = execute_tool(name, inputs)

        print(f"  {Style.DIM}Output: {result[:150]}...{Style.RESET_ALL}")

        # ToolMessage ties the result back to its originating tool call via tool_call_id
        tool_results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return {"messages": tool_results}


def should_continue(state: AgentState) -> str:
    if state["iteration_count"] >= MAX_ITERATIONS:
        print(f"\n  Max iterations ({MAX_ITERATIONS}) reached — sending to critic for failure diagnosis.")
        # route to critic so its RETRY reason documents what was missing, not a silent hard stop
        return "critic_node"
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tools"
    # no tool calls — hand off to critic before declaring done
    return "critic_node"


def critic_should_continue(state: AgentState) -> str:
    if state["critic_feedback"].upper().startswith("APPROVED"):
        return END
    # cap already hit — don't loop back into planner, just end with the critic's reason visible
    if state["iteration_count"] >= MAX_ITERATIONS:
        return END
    return "planner_node"


workflow = StateGraph(AgentState)
workflow.add_node("planner_node", planner_node)
workflow.add_node("call_model", call_model)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("critic_node", critic_node)
workflow.set_entry_point("planner_node")
workflow.add_edge("planner_node", "call_model")
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {"execute_tools": "execute_tools", "critic_node": "critic_node", END: END},
)
workflow.add_edge("execute_tools", "call_model")
workflow.add_conditional_edges(
    "critic_node",
    critic_should_continue,
    {"planner_node": "planner_node", END: END},
)
graph = workflow.compile()

# draw_mermaid_png calls the mermaid.ink API online — requires internet, no local deps
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("aria_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved → aria_graph.png")
except Exception as e:
    print(f"Could not render graph image ({e}); mermaid source:")
    print(graph.get_graph().draw_mermaid())


def run_agent(question: str) -> str:
    print(f"\n{Style.BRIGHT}Question: {question}{Style.RESET_ALL}")
    print("-" * 50)

    agent_start = time.time()

    initial_state: AgentState = {
        # system prompt lives in messages so it accumulates naturally with the rest of the history
        "messages": [SystemMessage(content=SYSTEM), HumanMessage(content=question)],
        "iteration_count": 0,
        "plan": [],
        "critic_feedback": "",
    }

    final_state = graph.invoke(initial_state)

    elapsed = time.time() - agent_start

    # walk backwards to find the last AI message that has actual text content
    last_ai = next(
        (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and m.content),
        None,
    )
    answer = last_ai.content if last_ai else "No answer produced."
    print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL} {answer}")
    print(f"\n  {Style.DIM}Total time: {elapsed:.1f}s | Iterations: {final_state['iteration_count']}{Style.RESET_ALL}")
    return answer


if __name__ == "__main__":
    run_agent("Find Mistral's most recently released model, go to their official announcement or blog post for benchmark numbers, and save a structured summary to mistral_research.md.")
