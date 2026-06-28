import os
import re
import sys
import time
import datetime
import anthropic
import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from typing import Annotated
from operator import add
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from tavily import TavilyClient
from langgraph.prebuilt import ToolNode


class Finding(BaseModel):
    fact: str = Field(description="One discrete, verifiable factual statement")
    url: str = Field(description="The source URL from the web searches that supports this fact")


class Evidence(BaseModel):
    findings: list[Finding] = Field(description="Distinct findings, each tied to a source URL")


class FlowState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    draft: str
    next: str
    iteration: int
    trace: list[str]
    has_research: bool
    fc_attempts: int
    sources: Annotated[list[dict], add]
    node_events: Annotated[list[str], add]


load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, api_key=os.getenv("CLAUDE_API_KEY"))

# Social / user-generated domains. Excluded from search results AND rejected as provenance by
# default — the stress battery showed they are junk provenance for ordinary factual blogs. The
# `allow_social` opt-in flag (see RUN_OPTS / run_blog_crew) flips both behaviours for topics where
# the social post IS the primary source (e.g. "what X's latest post signals").
SOCIAL_DOMAINS = [
    "facebook.com", "x.com", "twitter.com", "reddit.com", "linkedin.com",
    "instagram.com", "tiktok.com", "quora.com", "medium.com",
]

# Per-run options read by the web_search tool and the node prompt-builders. run_blog_crew() sets this
# before invoking the graph. A module-level global is fine for this single-run, single-threaded CLI;
# if this is ever run concurrently, switch to LangGraph config injection (RunnableConfig).
RUN_OPTS = {"allow_social": False}

# Publisher (dev.to). Drafts are uploaded UNPUBLISHED; the human reviews/publishes on dev.to.
DEVTO_URL = "https://dev.to/api/articles"
PUBLISH_TAGS = ["ai", "writing"]  # lowercase-alphanumeric, <=4 — dev.to 422s on invalid tags


@tool
def web_search(query: str) -> str:
    """Search the web for information on a given query."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    kwargs = {
        "search_depth": "advanced",  # highest relevance; better evidence up front -> fewer FC re-runs
        "max_results": 8,            # more evidence per call than the default 5
        # NOTE: include_raw_content is intentionally OFF — it returns full page bodies, and 8 results
        # x several parallel searches overflowed the 200k context window. "advanced" already returns
        # the most relevant chunks per source, which is the bounded richer-evidence win we want.
    }
    if not RUN_OPTS["allow_social"]:
        kwargs["exclude_domains"] = SOCIAL_DOMAINS
    results = client.search(query, **kwargs)
    return results


tool_node = ToolNode([web_search])

researcher_llm = llm.bind_tools([web_search])
fact_checker_llm = llm.bind_tools([web_search])
writer_llm = llm
supervisor_llm = llm


@retry(
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
    )),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def llm_invoke(model, messages):
    return model.invoke(messages)


def _provenance_rule() -> str:
    """Researcher rule 4 — flips with the allow_social opt-in flag."""
    if RUN_OPTS["allow_social"]:
        return (
            "4) This topic may center on social-media content, so social-media / user-generated URLs "
            "(x.com/twitter.com, facebook.com, reddit.com, linkedin.com, etc.) ARE acceptable PRIMARY "
            "sources here — cite the actual post/source URL the fact came from."
        )
    return (
        "4) Prefer authoritative sources (official reports, government/agency data, established news, "
        "research orgs). Do NOT rely on social-media or user-generated URLs (facebook.com, x.com/twitter.com, "
        "reddit.com, linkedin.com posts, forums) as the provenance for a fact — if a claim is real, find it "
        "on a primary or reputable secondary source instead."
    )


def _extraction_social_clause() -> str:
    """Extraction-step social handling — flips with the allow_social opt-in flag."""
    if RUN_OPTS["allow_social"]:
        return (
            "Social-media / user-generated URLs ARE acceptable sources for this topic — keep findings "
            "attributed to them. "
        )
    return (
        "Omit any finding whose only source is a social-media or user-generated URL (facebook.com, "
        "x.com/twitter.com, reddit.com, linkedin.com posts, forums) — keep a finding only if it is "
        "attributable to an authoritative source. "
    )


def researcher_node(state: FlowState) -> dict:
    print("\n[researcher] starting...")
    t0 = time.perf_counter()
    messages = [
        SystemMessage(content=(
            "You are a Research Agent. Your only job is to search the web using the web_search tool "
            "and gather relevant information about the given topic. "
            "Rules: "
            "1) Never answer from your own knowledge — always use web_search. "
            "2) Perform multiple searches if one query is not enough to cover the topic thoroughly. "
            "3) Your findings will be handed to a Writer agent that cites every claim, so each finding "
            "   must be a discrete, verifiable fact tied to the exact source URL it came from. "
            + _provenance_rule()
        )),
        HumanMessage(content=f"Research this topic thoroughly: {state['topic']}"),
        *state["messages"],
    ]
    for i in range(4):                 # ceiling lowered 5->4: clips runaway breadth on bare topics
        response = llm_invoke(researcher_llm, messages)
        if not response.tool_calls:
            break
        print(f"  [researcher] tool call {i + 1}: {[tc['name'] for tc in response.tool_calls]}")
        tool_results = tool_node.invoke({"messages": [response]})
        messages.append(response)
        messages.extend(tool_results["messages"])

    # Structured extraction: turn the gathered search results into discrete fact→url findings.
    # Messages end with ToolMessages (or the initial human turn), so there is no dangling tool_use.
    extraction_instruction = HumanMessage(content=(
        "Based ONLY on the web search results above, list up to ~30 of the most relevant, distinct "
        "factual findings relevant to the topic, each tied to the exact source URL it came from. "
        "Consolidate duplicate facts (if many sources state the same fact, keep it once with the best "
        "source) and prioritize the most important and verifiable; omit minor or redundant ones. "
        "Omit anything you cannot attribute to one of the URLs returned by your searches. "
        + _extraction_social_clause() +
        "Do not invent facts or URLs."
    ))
    structured_llm = llm.with_structured_output(Evidence)
    evidence = llm_invoke(structured_llm, messages + [extraction_instruction])

    # Stable global numbering: offset new ids past any evidence from an earlier researcher pass.
    start_id = len(state.get("sources", [])) + 1
    new_sources = [
        {"id": start_id + i, "fact": f.fact, "url": f.url}
        for i, f in enumerate(evidence.findings)
    ]
    rendered = "\n".join(f"[{s['id']}] {s['fact']} — {s['url']}" for s in new_sources)

    elapsed = time.perf_counter() - t0
    print(f"[researcher] done in {elapsed:.1f}s | {len(new_sources)} findings collected")
    return {
        "messages": [AIMessage(content=f"Research findings:\n{rendered}")],
        "has_research": True,
        "sources": new_sources,
        "node_events": [f"[researcher] {len(new_sources)} findings | {elapsed:.1f}s"],
    }


def render_references(draft: str, sources: list[dict]) -> tuple[str, dict]:
    """Renumber inline [n] markers to reading order, dedup by URL, and append the References list.

    The writer cites raw evidence-store ids; this maps each to a reader-facing number (one per unique
    URL, in first-citation order). A marker whose id is not in the store is a fabricated citation —
    it is rewritten to [uncited] (honest) and counted. Returns (draft, stats).
    """
    by_id = {s["id"]: s["url"] for s in sources}
    url_to_num: dict[str, int] = {}
    order: list[str] = []
    hallucinated = 0

    def repl(m):
        nonlocal hallucinated
        url = by_id.get(int(m.group(1)))
        if url is None:                       # cite resolves to nothing → honest fallback
            hallucinated += 1
            return "[uncited]"
        if url not in url_to_num:             # first sighting of this URL → next number
            order.append(url)
            url_to_num[url] = len(order)
        return f"[{url_to_num[url]}]"

    new = re.sub(r"\[(\d+)\]", repl, draft)          # renumber + dedup inline markers
    new = re.sub(r"(\[\d+\])\1+", r"\1", new)        # collapse adjacent dups: [1][1] -> [1]
    uncited = len(re.findall(r"\[uncited\]", new, flags=re.IGNORECASE))  # incl. converted ones
    lines = [f"[{url_to_num[u]}] {u}" for u in order]
    refs = "\n\n## References\n" + "\n".join(lines) if lines else ""
    return new + refs, {"sources": len(order), "uncited": uncited, "hallucinated": hallucinated}


def writer_node(state: FlowState) -> dict:
    print("\n[writer] starting...")
    t0 = time.perf_counter()
    evidence_block = "\n".join(f"[{s['id']}] {s['fact']}" for s in state["sources"])
    messages = [
        SystemMessage(content=(
            "You are a Writer Agent for a research assistant. You write flowing, readable prose, but you "
            "are citation-constrained: you assert ONLY what the evidence supports, and you attach an inline "
            "marker to every factual sentence. "
            "Rules: "
            "1) Place the [n] marker(s) at the very END of the factual sentence they support — immediately "
            "   AFTER the closing period (e.g. 'Solar capacity grew 40% in 2024. [3]'). NEVER put the marker "
            "   at the start of a sentence or in front of the fact it cites. "
            "   Use only ids that appear in the EVIDENCE block. Never renumber, never invent an id. "
            "2) Assert only what the evidence supports. For a necessary connective or contextual sentence "
            "   that no finding backs, mark it [uncited] — never attach a fabricated number to it. "
            "3) Do NOT write a References or Sources section, and never write a raw URL. The system appends "
            "   references automatically from your [n] markers. "
            "4) If fact-checker feedback is present in the conversation, apply the EXACT corrections: find the "
            "   CLAIM text in your draft and replace it with the CORRECTION value. Each correction has been "
            "   added to the EVIDENCE block as a new entry — cite the corrected sentence with that entry's "
            "   [n] id rather than leaving it uncited. "
            "5) Aim for 200-400 words: an engaging title, then smooth prose. Write for a general but curious "
            "   audience — readable, but every claim earns its citation."
        )),
        *state["messages"],
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"EVIDENCE (cite by number):\n{evidence_block}\n\n"
            "Write the piece now, citing each factual sentence with its [n] marker."
        )),
    ]
    response = llm_invoke(writer_llm, messages)
    body = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
    draft, stats = render_references(body, state["sources"])
    note = f"{stats['sources']} sources / {stats['uncited']} uncited"
    if stats["hallucinated"]:
        note += f" / {stats['hallucinated']} hallucinated"
    elapsed = time.perf_counter() - t0
    print(f"[writer] done in {elapsed:.1f}s | {note}")
    return {
        "draft": draft,
        "messages": [response, AIMessage(content="Draft complete.")],
        "node_events": [f"[writer] {note} | {elapsed:.1f}s"],
    }


def fact_checker_node(state: FlowState) -> dict:
    print("\n[fact_checker] starting...")
    t0 = time.perf_counter()
    messages = [
        SystemMessage(content=(
            "You are a Fact-Checker Agent. Your only job is to verify factual accuracy of the blog draft below. "
            "Rules: "
            "1) Always use web_search to verify claims — never judge based on your own knowledge. "
            "2) Only flag FACTUAL errors: wrong names, dates, statistics, or events. "
            "   Do NOT flag style, tone, word choice, or structure — those are not your job. "
            "3) Be proportionate: minor imprecision is acceptable. Only flag errors that would actively mislead a reader. "
            "   The draft contains [n] / [uncited] markers and a References list — these are citation annotations, "
            "   not claims. Ignore them; verify only the factual prose. "
            "4) CONTESTED FACTS: if your searches turn up reputable sources that genuinely DISAGREE on a value "
            "   (e.g. one source says August, another says September), the draft is NOT wrong — it is contested. "
            "   Do NOT flag a claim as an error merely because some source states a different value, as long as the "
            "   draft's figure is supported by at least one reputable source. Accept it and move on; flag only claims "
            "   that NO reputable source supports. This prevents flip-flopping a value back and forth across passes. "
            "5) BE EXHAUSTIVE: verify EVERY factual claim in the draft and report ALL errors you find in this single "
            "   pass — do NOT stop at the first error. You only get one revision round, so a claim you skip now will "
            "   ship uncorrected. Emit one CLAIM/CORRECTION/SOURCE triple for each distinct error. "
            "6) Your response MUST begin with exactly 'NO ISSUES FOUND' or 'ISSUES FOUND:' — no preamble, no introduction, no explanation before it. "
            "   Use one of these two formats only: "
            "   NO ISSUES FOUND "
            "   ISSUES FOUND:\n"
            "   CLAIM: <exact phrase from the draft that is wrong>\n"
            "   CORRECTION: <the verified correct value>\n"
            "   SOURCE: <url from your web search>\n"
            "   (repeat CLAIM/CORRECTION/SOURCE for each issue)"
        )),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"Draft:\n{state['draft']}"
        )),
    ]
    for i in range(8):
        response = llm_invoke(fact_checker_llm, messages)
        if not response.tool_calls:
            break
        print(f"  [fact_checker] tool call {i + 1}: {[tc['name'] for tc in response.tool_calls]}")
        tool_results = tool_node.invoke({"messages": [response]})
        messages.append(response)
        messages.extend(tool_results["messages"])
    # If cap hit with tool_calls still pending, force a clean, correctly-formatted verdict. Without an
    # explicit instruction the model just narrates its last search ("Perfect! I found the figure...")
    # and the supervisor can't recognize it as a verdict — re-running the whole node.
    if response.tool_calls:
        messages.append(HumanMessage(content=(
            "You have gathered enough information — stop searching. Output ONLY your final verdict now, "
            "beginning with exactly 'NO ISSUES FOUND' or 'ISSUES FOUND:' and following the required "
            "CLAIM / CORRECTION / SOURCE format. No preamble."
        )))
        response = llm_invoke(llm, messages)
    verdict = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

    # Promote each CORRECTION/SOURCE the fact-checker found into the evidence store so the writer can
    # cite the fix with a real [n] marker instead of leaving the corrected sentence [uncited].
    # "ISSUES FOUND" is a substring of "NO ISSUES FOUND" — and the FC sometimes emits a verdict that
    # carries both (a self-retracted issue, then NO ISSUES FOUND). Match the supervisor's routing
    # precedence (NO-ISSUES wins) so promotion only fires on a genuine issues verdict.
    new_sources: list[dict] = []
    if "ISSUES FOUND" in verdict and "NO ISSUES FOUND" not in verdict:
        extraction = HumanMessage(content=(
            "From the fact-check report below, extract each CORRECTION as a verified fact paired with "
            "its SOURCE url. Return only corrections that have a source.\n\n" + verdict
        ))
        corrections = llm_invoke(llm.with_structured_output(Evidence), [extraction])
        start_id = len(state["sources"]) + 1
        new_sources = [
            {"id": start_id + i, "fact": f.fact, "url": f.url}
            for i, f in enumerate(corrections.findings)
        ]

    elapsed = time.perf_counter() - t0
    print(f"[fact_checker] done in {elapsed:.1f}s | verdict: {verdict[:60].strip()}")
    return {
        "messages": [response],
        "sources": new_sources,
        "node_events": [f"[fact_checker] {elapsed:.1f}s | {verdict.strip()}"],
    }


def supervisor_node(state: FlowState) -> dict:
    iteration = state["iteration"] + 1

    if iteration > 9:
        trace_entry = f"iteration {iteration}: hard stop, iteration cap reached"
        return {
            "next": "end",
            "iteration": iteration,
            "trace": state["trace"] + [trace_entry],
        }

    draft = state["draft"]
    msgs = state["messages"]
    has_research = state["has_research"]

    def extract_text(msg) -> str:
        c = msg.content
        if isinstance(c, list):
            return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
        return str(c)

    latest_content = extract_text(msgs[-1]).strip() if msgs else ""
    is_fc_report = "NO ISSUES FOUND" in latest_content or "ISSUES FOUND" in latest_content

    new_fc_attempts = state["fc_attempts"]

    # --- DETERMINISTIC SPINE ---
    # All transitions here are rule-based; no LLM call needed.
    if not draft and not has_research:
        decision = "researcher"
        reason = "no research and no draft yet"

    elif has_research and not draft:
        decision = "writer"
        reason = "research ready, draft is empty"

    elif draft and not is_fc_report:
        decision = "fact_checker"
        reason = "draft exists, not yet fact-checked"

    elif "NO ISSUES FOUND" in latest_content:
        decision = "publisher"
        reason = "fact-checker found no issues — publishing"

    # --- ISSUES FOUND ---
    # The fact-checker is a flag-and-correct checker: it has already searched and emitted
    # CLAIM/CORRECTION/SOURCE triples (now added to state["sources"]), so the writer always has
    # what it needs to fix the draft — routing back to the researcher would only re-gather research
    # that already exists. Early failures: deterministically hand to the writer. On the third failure
    # (fc_attempts >= 3), a genuine agentic severity gate decides whether the residual issues are acceptable.
    else:
        new_fc_attempts = state["fc_attempts"] + 1
        if new_fc_attempts >= 3:
            severity_msgs = [
                SystemMessage(content=(
                    "The fact-checker still finds issues after 3 revision passes. "
                    "Read the remaining issues and classify them: "
                    "'MINOR' if they are acceptable imprecisions that would not actively mislead a reader. "
                    "'CRITICAL' if they are wrong facts a reader would rely on. "
                    "Treat as MINOR: contested values where the draft's figure is supported by at least one "
                    "reputable source, and small numeric/benchmark discrepancies (e.g. a latency or percentage "
                    "that differs slightly between sources). Reserve CRITICAL for claims that NO reputable source "
                    "supports, or that would materially mislead a reader. "
                    "Respond with ONLY one word: MINOR or CRITICAL."
                )),
                HumanMessage(content=latest_content),
            ]
            severity = llm_invoke(supervisor_llm, severity_msgs).content.strip().upper()
            if "CRITICAL" in severity:
                decision = "end"
                reason = f"FAILED CONVERGENCE — fc_attempts={new_fc_attempts}, remaining issues classified CRITICAL"
            else:
                decision = "publisher"
                reason = f"soft-accept — fc_attempts={new_fc_attempts}, remaining issues classified MINOR — publishing"
        else:
            decision = "writer"
            reason = "fact-checker found issues; writer applying CORRECTION/SOURCE fixes"

    trace_entry = f"iteration {iteration}: routing to {decision} — {reason}"
    print(f"[supervisor] {trace_entry}")
    return {
        "next": decision,
        "iteration": iteration,
        "trace": state["trace"] + [trace_entry],
        "fc_attempts": new_fc_attempts,
    }


def _extract_title_and_body(draft: str, topic: str) -> tuple[str, str]:
    """Title = first '# ' heading (dev.to renders the title field separately, so strip that heading
    from the body to avoid a duplicate). Fallback to the topic when the draft has no '# ' heading."""
    lines = draft.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^#\s+", line):
            title = re.sub(r"^#\s+", "", line).strip()
            body = "\n".join(lines[:i] + lines[i + 1:]).lstrip("\n")
            return title, body
    return topic, draft


def publisher_node(state: FlowState) -> dict:
    """Terminal node: upload the final draft to dev.to as an UNPUBLISHED draft. Never raises —
    any failure (missing key, network, 4xx) is logged and the graph still ends cleanly."""
    print("\n[publisher] starting...")
    key = os.getenv("DEV_API_KEY")
    if not key:
        msg = "[publisher] DEV_API_KEY not set — skipping upload"
        print(msg)
        return {"node_events": [msg]}

    title, body = _extract_title_and_body(state["draft"], state["topic"])
    payload = {"article": {
        "title": title,
        "body_markdown": body,
        "tags": PUBLISH_TAGS,
        "published": False,
    }}
    try:
        resp = requests.post(
            DEVTO_URL,
            headers={"api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
    except requests.RequestException as e:
        msg = f"[publisher] request failed: {type(e).__name__}: {e}"
        print(msg)
        return {"node_events": [msg]}

    if resp.status_code == 201:
        url = resp.json().get("url", "(no url in response)")
        msg = f"[publisher] draft created: {url}"
    else:
        msg = f"[publisher] failed ({resp.status_code}): {resp.text[:300]}"
    print(msg)
    return {"node_events": [msg]}


def route(state: FlowState) -> str:
    return state["next"]


graph = StateGraph(FlowState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("fact_checker", fact_checker_node)
graph.add_node("publisher", publisher_node)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher": "researcher",
        "writer":     "writer",
        "fact_checker": "fact_checker",
        "publisher":  "publisher",
        "end":        END,
    },
)

graph.add_edge("researcher",   "supervisor")
graph.add_edge("writer",       "supervisor")
graph.add_edge("fact_checker", "supervisor")
graph.add_edge("publisher",    END)

app = graph.compile()


def log_run(topic: str, result: dict, latency: float) -> None:
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.md")

    run_number = 1
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            run_number = f.read().count("## Run ") + 1

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace_block = "\n".join(result["trace"])
    node_events_block = "\n".join(result.get("node_events", []))
    sources_count = len(result.get("sources", []))

    entry = (
        f"## Run {run_number} — {timestamp}\n"
        f"**Topic:** {topic}\n"
        f"**Latency:** {latency:.2f}s\n"
        f"**Supervisor passes:** {result['iteration']}\n"
        f"**Sources collected:** {sources_count}\n\n"
        f"**Node events:**\n{node_events_block}\n\n"
        f"**Routing trace:**\n{trace_block}\n\n"
        f"**Final draft:**\n{result['draft']}\n\n"
        f"---\n"
    )

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(entry)


def run_blog_crew(topic: str, allow_social: bool = False) -> dict:
    # Gates both the search-level exclude_domains (web_search tool) and the provenance prompts
    # (_provenance_rule / _extraction_social_clause). Default off; opt in for social-as-subject topics.
    RUN_OPTS["allow_social"] = allow_social
    initial_state = {
        "messages":      [],
        "topic":         topic,
        "draft":         "",
        "next":          "",
        "iteration":     0,
        "trace":         [],
        "has_research":  False,
        "fc_attempts":   0,
        "sources":       [],
        "node_events":   [],
    }
    t0 = time.perf_counter()
    result = app.invoke(initial_state, config={"recursion_limit": 25})
    log_run(topic, result, time.perf_counter() - t0)
    return result


if __name__ == "__main__":
    final = run_blog_crew("The history of Sandwich", allow_social=True)

    print("\n" + "=" * 60)
    print("FINAL DRAFT")
    print("=" * 60)
    print(final["draft"])

    print("\n" + "=" * 60)
    print("ROUTING TRACE")
    print("=" * 60)
    for entry in final["trace"]:
        print(entry)
