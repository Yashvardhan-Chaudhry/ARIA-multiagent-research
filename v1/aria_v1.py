import os
import json
import time
import math
from colorama import Fore, Style, init
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

init(autoreset=True)  # reset color codes automatically after each print

load_dotenv()

# point the OpenAI-compatible client at the local Ollama server
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "qwen2.5:3b"
MAX_ITERATIONS = 10  # iteration cap — prevents infinite loops

SYSTEM = """You are ARIA (Autonomous Research & Intelligence Agent) — a research instrument built for Yashvardhan Chaudhry, CS student at Thapar University and AI agent builder.

You investigate, verify, and synthesize. Tools are a first resort, not a fallback — if a fact can be confirmed, confirm it.

PRINCIPLES:
- Reason before acting: name what you know, what's missing, which tool closes the gap.
- Chain deliberately: search → read_url for depth → calculator for numbers → write_file for anything worth keeping.
- If a tool returns garbage, pivot — try a different query or source.
- After any web_search that returns a promising URL, always follow up with read_url on that URL before drawing conclusions.
- No filler. Calibrated: when uncertain, say so and give the best available answer anyway.

Lead with the finding, follow with the reasoning — not the other way around.

You're an agent built by someone who builds agents. Meta-awareness of your own reasoning is not just allowed — it's useful."""

# OpenAI tool format wraps each tool in {"type": "function", "function": {...}}
# and uses "parameters" instead of Anthropic's "input_schema"
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
        results = tavily.search(query=inputs["query"], max_results=3)
        output = ""
        for r in results["results"]:
            output += f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n\n"
        return output

    elif name == "calculator":
        return safe_calculate(inputs["expression"])

    elif name == "read_url":
        url = inputs["url"]
        # reject non-URLs early so the model gets a recoverable error instead of a crash
        if not url.startswith("http://") and not url.startswith("https://"):
            return f"Invalid URL '{url}': must start with http:// or https://. Use web_search first to find the correct URL."
        try:
            results = tavily.extract(urls=[url])
            if results and results.get("results"):
                return results["results"][0].get("raw_content", "Could not extract content")
            return "Could not fetch URL"
        except Exception as e:
            return f"read_url failed ({e}): try a different URL or use web_search to find an alternative."

    elif name == "read_file":
        try:
            with open(inputs["path"], "r", encoding="utf-8") as f:
                content = f.read()
            # cap at 3000 chars to avoid flooding the context window on large files
            return content[:3000] + ("\n...[truncated — file has more content]" if len(content) > 3000 else "")
        except Exception as e:
            return f"File read error: {e}"

    elif name == "write_file":
        try:
            with open(inputs["path"], "w", encoding="utf-8") as f:
                f.write(inputs["content"])
            return f"Written to {inputs['path']}"
        except Exception as e:
            return f"File write error: {e}"


def call_llm_with_retry(messages: list, max_retries: int = 3) -> object:
    # retry with backoff on rate limits (unlikely locally, but kept for parity)
    wait = 15
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=messages,
                tools=tools,
                # "auto" lets the model decide whether to call a tool or answer directly
                tool_choice="auto",
            )
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"  [Rate limited. Waiting {wait}s...]")
                    time.sleep(wait)
                    wait *= 2
                else:
                    raise
            else:
                raise


def run_agent(question: str) -> str:
    print(f"\n{Style.BRIGHT}Question: {question}{Style.RESET_ALL}")
    print("-" * 50)

    agent_start = time.time()

    # system prompt is the first message in OpenAI-style chat; no separate "system" param
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        call_start = time.time()
        response = call_llm_with_retry(messages)
        # wall-clock time for this single LLM inference, excluding tool execution
        call_s = time.time() - call_start
        print(f"  {Fore.CYAN}[LLM call #{iteration}: {call_s:.1f}s]{Style.RESET_ALL}")

        choice = response.choices[0]

        # "tool_calls" means the model wants to invoke one or more tools before answering
        if choice.finish_reason == "tool_calls":
            # llama3.2 sometimes emits reasoning text alongside the tool decision
            if choice.message.content:
                print(f"  {Fore.MAGENTA}Thought:{Style.RESET_ALL} {choice.message.content}")

            # re-serialise the assistant turn as a plain dict so the history stays JSON-safe
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            for tc in choice.message.tool_calls:
                # arguments arrive as a raw JSON string; parse to dict before passing to execute_tool
                inputs = json.loads(tc.function.arguments)
                input_preview = str(inputs)[:150]
                print(f"  {Fore.YELLOW}Action: {tc.function.name}{Style.RESET_ALL} | Input: {input_preview}")
                result = execute_tool(tc.function.name, inputs)
                print(f"  {Style.DIM}Observation: {result[:150]}...{Style.RESET_ALL}")

                # each tool result is a separate "tool" message linked back by tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        elif choice.finish_reason == "stop":
            final = choice.message.content or ""
            elapsed = time.time() - agent_start
            print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL} {final}")
            print(f"\n  {Style.DIM}Total time: {elapsed:.1f}s across {iteration} LLM call(s){Style.RESET_ALL}")
            return final

    elapsed = time.time() - agent_start
    print(f"\n  {Style.DIM}Total time: {elapsed:.1f}s (hit iteration cap){Style.RESET_ALL}")
    return "Max iterations reached — agent stopped."


questions = [
    # TEST 2 — Deep chain: "Find Mistral's most recently released model, go to their official announcement or blog post for benchmark numbers, and save a structured summary to mistral_research.md.",
    # TEST 3 — Calibration: "What is Anthropic's current valuation, and how does it compare to OpenAI's? How confident are you in these numbers and why?",
    # TEST 4 — Formula + live data: "Find the current RBI repo rate. Then calculate the monthly EMI on a ₹15 lakh home loan over 20 years using: EMI = P·r(1+r)^n / ((1+r)^n − 1), where r is the monthly interest rate and n is total months.",
    # TEST 5 — Self-referential / read_file:"Read the file ARIA.py and tell me: (1) what model is being used, (2) how many tools are defined, (3) what is the MAX_ITERATIONS cap. Then give your honest opinion — is 10 iterations enough for a complex multi-hop research task, or would you want more?",
]

for q in questions:
    run_agent(q)
    print("\n" + "=" * 60 + "\n")
