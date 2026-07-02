# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

This repo is **ARIA** — an autonomous research agent kept in three versions side by side (`v1/`, `v2/`, `v3/`) to show its evolution. **v3 is the flagship**; v1/v2 are earlier, intentionally rougher versions whose documented failures motivated v3's design. See `README.md` for the narrative.

> Split out of the original `async-batch` monorepo on 2026-06-28 as a standalone portfolio repo. This is now the **working copy** for ARIA — edit and commit here.

## Running Scripts

Uses [`uv`](https://docs.astral.sh/uv/) (fetches Python 3.14 automatically):

```bash
uv sync                                       # install deps into .venv
cp .env.example .env                          # then fill in real keys

uv run python v3/aria_v3_multiagent.py        # v3 crew — topic set in __main__ block
uv run python v3/Tavily_Research_blackbox.py  # optional A/B vs Tavily /research

ollama pull qwen2.5:3b                         # v1/v2 run on a LOCAL model
uv run python v1/aria_v1.py                    # or v2/aria_v2.py
```

## Environment Variables

All keys live in `.env` (gitignored; template in `.env.example`).

- **`CLAUDE_API_KEY`** — Anthropic key; powers **v3** (`ChatAnthropic`, `claude-haiku-4-5-20251001`).
- `TAVILY_API_KEY` — web search for the `web_search` tool (v1 + v3 researcher/fact_checker).
- `TAVILY_API_KEY2` — separate key for the Tavily `/research` one-shot in the A/B harness (`v3/Tavily_Research_blackbox.py`), so its credit burn is isolated from the crew.
- `DEV_API_KEY` — dev.to key for the v3 publisher node.
- **v1 and v2 need NO API key for the model** — they run on local Ollama (`qwen2.5:3b`) via an OpenAI-compatible endpoint at `http://localhost:11434`. (v1's `web_search` still uses `TAVILY_API_KEY`.)

## Repository Layout

- `v1/aria_v1.py` — Bare tool-calling agentic loop, local Ollama. Tools: web_search, calculator, read_url, read_file, write_file. Manual tool dispatch + retry. Test log: `v1/aria_testing.md`.
- `v2/aria_v2.py` — LangGraph `StateGraph`: planner → call_model → execute_tools → critic. Critic validates completeness before terminating. Local Ollama. Test log: `v2/aria_v2_testing.md`.
- `v3/aria_v3_multiagent.py` — the multi-agent crew (details below).
- `v3/eval_report.md` — auto-appended stress-run log (14+ runs). `v3/Blackbox_vs_crew.md` — A/B vs Tavily.

## Architecture: ARIA v3 — the crew

LangGraph cited-research crew: **researcher → writer → fact_checker → supervisor**, with a terminal `publisher` node on clean termination (uploads an unpublished dev.to draft). The writer is a **citation-constrained synthesizer**, not a free-prose blogger: it asserts only what the evidence supports and tags every factual sentence with a source. Key design details:

- `FlowState` (TypedDict) carries: `messages` (add_messages reducer), `topic`, `draft`, `next`, `iteration`, `trace`, `has_research`, `fc_attempts`, `sources` (Annotated[list[dict], add] — `{id, fact, url}` evidence entries accumulated across researcher AND fact_checker runs), `node_events` (Annotated[list[str], add] — per-node timing/verdict).
- **Evidence provenance**: the researcher runs its tool loop, then a `with_structured_output(Evidence)` call (Pydantic `Finding`/`Evidence` schemas) extracts discrete `{fact, url}` findings. Ids are assigned with an offset (`len(state["sources"]) + 1`) so re-runs never collide. Provenance keys to *facts*, not to the search query.
- **Citation pipeline**: the writer cites raw evidence ids inline (`[n]`) at the END of the factual sentence they support (trailing, not leading — explicit prompt rule + worked example), marks unsupported connective sentences `[uncited]`, and writes NO URLs/References itself. `render_references()` (pure, deterministic) renumbers markers to reading order (first cited = `[1]`), dedups by URL, collapses adjacent dups (`[1][1]`→`[1]`), and rewrites any hallucinated id to `[uncited]`. The `## References` list is built from the trusted store, so the writer cannot fabricate a URL.
- **Deterministic supervisor**: routine transitions are rule-based if/elif. On `ISSUES FOUND` it routes deterministically to `writer` (the fact_checker already supplies CORRECTION+SOURCE). The only LLM call is the third-failure severity gate.
- **`fc_attempts`**: incremented each `ISSUES FOUND`. Early failures route back to `writer`. On the **third** failure, a severity LLM call: `MINOR` → soft-accept + publish; `CRITICAL` → log `FAILED CONVERGENCE` + end. An `iteration > 9` hard stop backstops everything. Contested values (supported by ≥1 reputable source) and small numeric/benchmark discrepancies are ruled MINOR; CRITICAL is reserved for unsupported/materially-misleading claims.
- **Fact-checker is exhaustive + contested-fact aware**: reports ALL errors in one pass (multiple CLAIM/CORRECTION/SOURCE triples), and will NOT flag a value wrong when reputable sources genuinely disagree and the draft is supported by at least one — prevents writer↔FC flip-flopping. On issues, a `with_structured_output(Evidence)` call promotes each CORRECTION/SOURCE into `state["sources"]` so the writer can cite the fix with a real `[n]`.
- **Tool loop cap + cleanup**: researcher/fact_checker run internal tool loops. If a loop exits with `tool_calls` still set, a cleanup call to the base `llm` (no tools) forces a clean text response — avoids dangling `tool_use` blocks (Anthropic 400s). The fact_checker cleanup appends an explicit instruction to emit the verdict format.
- **Tavily `web_search` tuning**: `search_depth="advanced"`, `max_results=8`. Do NOT add `include_raw_content` — full page bodies × parallel searches overflow the 200k context window.
- **Source quality + `allow_social`**: social/UGC domains excluded at search level and rejected as provenance. `run_blog_crew(topic, allow_social=True)` opts back in for topics where the post IS the source. Default OFF.
- `llm_invoke()` wrapped in `@retry` (exponential backoff) for `RateLimitError`, `APIConnectionError`, `APITimeoutError`.
- Run results auto-append to `v3/eval_report.md` (node events, sources count, routing trace, final draft).

## Open follow-ups (revisit only on a real trigger)
- **File split** — `aria_v3_multiagent.py` is ~600 lines and stable; split into a package (`state.py`, `config.py`, `nodes.py`, `publisher.py`, `graph.py`, `runner.py`) only if it crosses ~700 lines or a new stateful subsystem is added.
- **Full deterministic anti-oscillation guard** — currently the `fc_attempts >= 3` severity gate + contested-fact rule bound the writer↔FC loop; a full prior-pass-similarity guard is deferred unless oscillation recurs frequently.
