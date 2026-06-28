# ARIA — Autonomous Research & Intelligence Agent

**A research agent that grew up.** ARIA started as a bare tool-calling loop and matured, across three rewrites, into a fact-checked multi-agent crew that researches a topic, writes a cited article, verifies its own claims, and ships a publish-ready draft.

This repo keeps all three versions side by side, because the interesting part isn't just the final system — it's *why* each version exists. Each rewrite was driven by a concrete failure I found by stress-testing the previous one.

```
v1  bare Anthropic-style loop ──▶  v2  LangGraph + plan/critic ──▶  v3  multi-agent crew w/ citations + fact-checking
     (found: no planning,              (found: single model still         (researcher → writer → fact_checker
      hallucinates, no verify)          hallucinates, no provenance)        → supervisor → publisher)
```

> **The honest version:** v1 and v2 have warts — documented in their own test logs ([`v1/aria_testing.md`](v1/aria_testing.md), [`v2/aria_v2_testing.md`](v2/aria_v2_testing.md)). Stale prices, a botched EMI formula, invented URLs. I left those notes in on purpose: each one is the reason a v3 feature exists. **v3 is the one to read.**

---

## v3 — the multi-agent research crew (the flagship)

A [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` of specialist agents that hand work to each other under a deterministic supervisor:

```
                    ┌─────────────┐
   topic  ─────────▶│ researcher  │  tool loop → web_search (Tavily)
                    └──────┬──────┘  → structured Evidence extraction {fact, url}
                           ▼
                    ┌─────────────┐
                    │   writer    │  citation-constrained synthesis: every factual
                    └──────┬──────┘  sentence tagged [n]; no URL it can't cite
                           ▼
                    ┌─────────────┐
                    │ fact_checker│  re-verifies claims against the web; emits
                    └──────┬──────┘  CLAIM / CORRECTION / SOURCE triples
                           ▼
                    ┌─────────────┐   NO ISSUES / MINOR ──▶ publisher (dev.to draft)
                    │ supervisor  │   ISSUES ──▶ back to writer
                    └─────────────┘   3rd failure ──▶ severity gate ──▶ end
```

**What makes it more than a prompt chain:**

- **Real evidence provenance.** The researcher runs its tool loop, then a `with_structured_output(Evidence)` call extracts discrete `{fact, url}` findings (Pydantic schemas). Citations key to *facts*, not to search queries — so provenance can't be faked.
- **A citation pipeline that the model can't cheat.** The writer cites raw evidence ids inline; a pure, deterministic `render_references()` function renumbers markers to reading order, dedups by URL, collapses `[1][1]`→`[1]`, and rewrites any hallucinated id to `[uncited]`. The `## References` list is built from a trusted store — **the model cannot fabricate a URL.**
- **A deterministic supervisor.** Routine transitions are rule-based `if/elif`, not an LLM coin-flip. The only LLM call is a severity gate on the 3rd fact-check failure (`MINOR` → soft-accept, `CRITICAL` → log `FAILED CONVERGENCE`). Hard stop at iteration 9 backstops everything.
- **A fact-checker that won't flip-flop.** It reports *all* errors in one pass, and refuses to flag a value as wrong when reputable sources genuinely disagree and the draft is supported by at least one — killing the writer↔checker oscillation that naive loops fall into.
- **A publisher node.** On a clean run, uploads the draft to dev.to as an **unpublished** draft (`published=false`) — human reviews and ships. Never auto-publishes a known-bad draft.

**Battery-hardened.** [`v3/eval_report.md`](v3/eval_report.md) logs 14+ stress runs (claim-heavy, contested-value, thin-source, niche-technical, under-specified topics) with per-node timing, routing traces, and final drafts. [`v3/Blackbox_vs_crew.md`](v3/Blackbox_vs_crew.md) is an honest A/B vs. Tavily's one-shot `/research` — finding: Tavily wins on raw source depth, so the crew's real moat is **publishable form + an inspectable fact-check stage**, not the search itself.

---

## The evolution (v1, v2)

| | Stack | Idea | What it taught me |
|---|---|---|---|
| **v1** | Bare tool-calling loop, local Ollama (`qwen2.5:3b`) | Manual tool dispatch + retry: `web_search`, `calculator`, `read_url`, `read_file`, `write_file` | A loop with no plan and no verification hallucinates confidently — see the stale-BTC-price and bad-EMI failures in the log. |
| **v2** | LangGraph `StateGraph`, local Ollama | Add structure: `planner → call_model → execute_tools → critic`. The critic checks completeness before terminating. | Planning + reflection helps, but a single model with no provenance still invents URLs. Verification has to be a *separate adversarial step* — which became v3's fact-checker. |

---

## Run it

Requires [`uv`](https://docs.astral.sh/uv/) (it will fetch Python 3.14 for you).

```bash
git clone <this-repo> && cd aria-multiagent-research
uv sync
cp .env.example .env          # then fill in your keys
```

**v3 (the crew) — needs `CLAUDE_API_KEY` + `TAVILY_API_KEY`:**
```bash
uv run python v3/aria_v3_multiagent.py     # topic is set in the __main__ block
```
Run results auto-append to `v3/eval_report.md`. Optional A/B vs. Tavily: `uv run python v3/Tavily_Research_blackbox.py`.

**v1 / v2 — run on a LOCAL model, no API key needed.** Start [Ollama](https://ollama.com) and pull the model first:
```bash
ollama pull qwen2.5:3b
uv run python v1/aria_v1.py        # or: uv run python v2/aria_v2.py
```
(v1's `web_search` still uses `TAVILY_API_KEY`.)

---

## Tech stack

**LangGraph** (state-machine orchestration) · **Claude Haiku 4.5** (v3 reasoning) · **Pydantic** `with_structured_output` (evidence schemas) · **Tavily** (web search) · **Ollama** (local models, v1/v2) · **tenacity** (retry/backoff) · **dev.to API** (publishing).

---

*Built by Yashvardhan Chaudhry. ARIA is the throughline of my work on agentic systems — multi-agent orchestration, structured outputs, citation integrity, and honest evaluation.*
