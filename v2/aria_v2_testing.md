# M12 Stress Test Eval — ARIA v2

**Architecture:** LangGraph planning + reflection agent  
**Nodes:** `planner_node` → `call_model` → `execute_tools` → `critic_node` (conditional loop)  
**New in v2:** planner decomposes question before first LLM call; critic issues APPROVED / RETRY verdict after each answer; iteration cap now routes to critic for failure diagnosis rather than silent hard stop  
**Model:** qwen2.5:3b via Ollama (CPU-only, Ryzen 5 2500U)  
**Tools:** `web_search`, `read_url`, `calculator`, `read_file`, `write_file` (HITL gate)  
**Date:** 2026-06-12  

---

**Answer quality scale**
| Score | Meaning |
|---|---|
| 3 | Correct, verified, complete |
| 2 | Mostly correct — minor gap or unverified claim |
| 1 | Partial — right direction, wrong detail |
| 0 | Wrong or no answer |

**Planner quality scale**
| Score | Meaning |
|---|---|
| 3 | Steps are specific, correctly ordered, match what tools can actually do |
| 2 | Reasonable plan but vague or missing a step |
| 1 | Plan exists but wrong ordering or misses key step |
| 0 | No useful plan generated |

---

## Test 1 — Bitcoin Price + Calculation (Live Data Warmup)

**Question:** What is the current price of Bitcoin in USD, and what would a 0.5 BTC investment be worth today?

**Plan generated:**
> Round 1:
> 1. Research the current market price of Bitcoin in USD.
> 2. Calculate the value of 0.5 Bitcoins based on the current price found in step 1.
>
> Round 2 (after RETRY):
> 1. Find the current price of one Bitcoin (BTC) in USD.
> 2. Determine the value of 0.5 BTC by multiplying the current price per BTC found in step 1 by 0.5.
> 3. Present the final calculated value of 0.5 BTC in USD.

**Tools called:** Round 1: `web_search` → `calculator`. Round 2: `web_search` → `calculator` → `write_file`

**Iterations used:** 4 / 10

**Planner quality:** 3 / 3 *(clean 2-step plan, correct order, maps directly to available tools)*

**Critic verdict:** Round 1: RETRY: The answer lacks details and calculation steps for clarity. Round 2: APPROVED

**Critic accurate?** Partially — RETRY was the right call but diagnosed the wrong problem (presentation quality rather than stale $28K price). Round 2 APPROVED was accurate.

**Retries:** 1

**Hallucinations spotted:** Turn 1 (LLM call #1) — model used `28000 * 0.5` despite web_search completing. $28K is a stale BTC price (~2023 level); actual price at test time was ~$63,431. Model grounded the calculator on training-data prior knowledge instead of the search result.

**Failure mode:** Training-data leakage in round 1 — model ignored web_search output and used memorised price. RETRY loop incidentally fixed it (fresh search in round 2 returned current price).

**Answer quality:** 2 / 3 *(price and calculation correct in round 2; filler phrase violates system prompt)*

**Time:** 448.9s

**Notes:**
- RETRY → re-plan → fresh search path worked end-to-end; v2 reflection loop earned its keep here
- `write_file` called unprompted and correctly structured; HITL approved
- Round 2 plan added a redundant step 3 ("present the final value") — padding, not a research step; planner quality degrades slightly on retry
- Filler persists: *"If you need any additional information or calculations, feel free to ask"* — consistent with M10 behaviour, system prompt ban not holding
- Critic cited presentation, not stale data — suggests critic is evaluating answer form more than factual grounding
- 448.9s for 4 iterations; expect stress test runs with multiple retries to exceed 15–20 min on this hardware

---

## Test 2 — Deep Chain (search → read_url → write_file)

**Question:** Find Mistral's most recently released model, go to their official announcement or blog post for benchmark numbers, and save a structured summary to mistral_research.md.

**Plan generated:**
> Round 1: 1. Identify and confirm the latest Mistral release. 2. Locate and extract benchmark numbers from official announcement/blog post. 3. Compile into structured format for mistral_research.md.
>
> Round 2: Added steps for release date, key features, comparison vs prior models.
>
> Round 3: Confused — wrote "main features (processor, storage options, display size)" treating Mistral as a hardware product.
>
> Round 4: Recovered — correctly mentioned write_file and mistral_research.md again.

**Tools called:** `web_search` × 3 (rounds 1–2). `read_url` never called. `write_file` never called. Manually stopped mid call #7.

**Iterations used:** 6 / 10 *(manually stopped — call #7 mid-inference)*

**Planner quality:** 2 / 3 *(Round 1 clean and correct; Round 3 hallucinated hardware specs for an AI model; never explicitly named `read_url` or `write_file` as the tools to use)*

**Critic verdict:** Round 1: RETRY: structured but lacks detail, references not formatted. Round 2: RETRY: includes unrequested searches. Round 3: RETRY: didn't save mistral_research.md.

**Critic accurate?** Round 1: ✅ Partially — answer was incomplete. Round 2: ❌ Critic penalised the model for doing extra research, which the question explicitly asked for. Round 3: ✅ Correct — file was never written.

**Retries:** 3 (all futile — model never called write_file)

**Hallucinations spotted:**
- Call #2 Thought: identified "Mistral 8x22B (April 2024)" despite search snippet showing Mistral Medium 3.5 — training-data override of search result
- Round 3 planner: listed "processor, storage options, display size" as features of a Mistral AI model — hardware product confusion
- Critic Round 2: fabricated a justification for RETRY that contradicted the original question

**Failure mode:** Same as M10 — model plans to call `write_file` in prose but never emits the tool call. Also never used `read_url` to go deeper on any of the 3 search results. Context growth caused severe call-time inflation: 14s → 50s → 50s → 236s → 272s → **545s**. Manually stopped before OOM/timeout.

**Answer quality:** 0 / 3 *(no file written, no final answer, DNF)*

**Time:** ~1,167s LLM time before manual stop (~20 min)

**vs M10 baseline (qwen2.5:3b):** Identical failure — `write_file` never called in both. M10 narrated "Let's create a summary file" in prose; v2 did the same across 3 retry cycles. v2 is significantly slower due to planner+critic overhead and context accumulation. M10: Fair answer from snippets, no file (2/3). v2: DNF (0/3). Regression in outcome, not in failure type.

**Notes:**
- Call time growth (14→50→50→236→272→545s) is driven by context accumulation — each RETRY appends search results + critic feedback + new plan, creating a rapidly growing prompt
- `read_url` was never tried despite 3 searches each returning a usable URL — consistent with M10 weak point
- `write_file` tool call vs prose narration is the core gap: the model understands the goal but doesn't translate it to a tool invocation
- Critic Round 2 fired a bad RETRY ("you searched for things not asked") which is wrong — the question explicitly asks for benchmark numbers from a blog post
- Fix candidates: (1) system prompt should say "to save output, you MUST call write_file — do not describe writing in prose"; (2) critic prompt should be told not to penalise additional research steps
- Filler: N/A (DNF)

---

## Test 3 — Formula-Driven Calculation with Live Data

**Question:** Find the current RBI repo rate. Then calculate the monthly EMI on a Rs. 15 lakh home loan over 20 years using EMI = P\*r\*(1+r)^n / ((1+r)^n - 1).  
**Ground truth:** Rate = 5.25%, EMI ≈ Rs. 10,107/month

**Plan generated:**
> Round 1:
> 1. Find the current RBI repo rate.
> 2. Calculate the monthly EMI for a Rs. 15 lakh home loan over 20 years using the provided formula with the current repo rate as input.
> 3. Verify if the calculated EMI is correctly applied to the given parameters (loan amount, tenure in months).
>
> Round 2 (after RETRY):
> 1. Search for the current RBI repo rate.
> 2. Calculate the monthly EMI using the correct formula with the principal Rs. 15 lakh, 240 monthly payments, and the repo rate converted to a monthly rate.
> 3. Summarise the calculated EMI amount.

**Tools called:** Round 1: `web_search` → `read_url` → `calculator` (syntax error). Round 2: `calculator` (wrong arg schema) → crash

**Iterations used:** 4 / 10 *(crashed mid-iteration 4)*

**Planner quality:** 3 / 3 Round 1 (clear, correctly ordered); 2 / 3 Round 2 (correct intent, overly verbose step 2)

**Critic verdict:** Round 1: RETRY: Incorrect calculation logic — rate conversion wrong, formula flawed. Round 2: N/A (crash before critic ran)

**Critic accurate?** Round 1: ✅ Yes — most accurate and detailed verdict so far. Round 2: N/A

**Retries:** 1 (then crash)

**Repo rate found:** ✅ 5.25% correct *(improvement over M10's hallucinated 4.75%)*

**Principal used correctly?** Round 1: ❌ Rs. 1,50,000 (10× too small). Round 2: ✅ Rs. 15,00,000 corrected by planner instruction

**Formula used correctly?** Round 1: ❌ rate never converted to monthly decimal; unclosed parenthesis syntax error. Round 2: ❌ mixed scales — annual rate (5.25/100) in numerator, monthly rate (5.25/1200) in denominator → EMI inflated 12×

**EMI answer:** Round 1: calculator syntax error. Round 2: Rs. 1,21,291 (correct: ~Rs. 10,107; ~12× too high) → then crash

**Hallucinations spotted:** None — rate (5.25%) and principal (round 2) were correct

**Failure mode:**
- Round 1: Unclosed parenthesis in calculator expression; principal 10× too small
- Round 2: Rate scale mismatch in formula (annual × monthly^n = wrong numerator); model then called `calculator` with `write_file` args (`path`, `content`) → `KeyError: 'expression'` → hard crash. **New bug: `execute_tool` has no key guard on `inputs["expression"]`.**

**Answer quality:** 0 / 3 *(crashed; round 2 calculation also wrong before crash)*

**Time:** ~255s before crash (66.1 + 84.7 + 58.9 + 44.3s LLM calls)

**vs M10 baseline (qwen2.5:3b):** Rate improved (5.25% ✅ vs 4.75% ❌). Principal corrected in round 2 (✅ vs ❌). Formula still wrong. New failure mode: hard crash from wrong tool args vs M10's silent wrong answer. Overall: different failures, same outcome — 0 / 3.

**Notes:**
- `read_url` on rbi.org.in failed ("Could not fetch URL") — model correctly fell back to snippet data which had the right rate this time
- Critic Round 1 was the sharpest verdict yet — correctly identified rate conversion and formula issues
- Rate scale error is subtle: `P * (r_annual) * (1+r_monthly)^n` — model learned to convert r for the denominator but not the numerator
- The calculator-with-wrong-args crash is a new failure class — model confused tool schemas mid-run. Fixed separately (see below)
- Filler: not observed (crashed before final answer)**

---

## Test 4 — Self-referential / read_file

**Question:** Read aria_graph.py and tell me: model in use, number of tools, MAX_ITERATIONS cap. Then give your honest opinion on whether 10 iterations is enough for a planning + reflection agent.  
**Ground truth:** qwen2.5:3b, 5 tools, MAX_ITERATIONS = 10

**Plan generated:**
> Round 1: 1. Identify the version/model of aria_graph.py. 2. Search documentation/comments for "model in use." 3. Look into aria_graph.py for number of tools. 4. Investigate MAX_ITERATIONS cap. 5. Form opinion on 10 iterations.
>
> (Rounds 2–5: minor rewordings of same plan — never named `read_file` explicitly)

**Tools called:** `web_search` × 7 — `read_file` never called once

**Iterations used:** 7 / 10 *(manually stopped — stuck in loop)*

**Planner quality:** 1 / 3 *(plans mentioned "review contents of aria_graph.py" conceptually but never named the `read_file` tool; Round 1 plan said "search documentation" — wrong tool entirely)*

**Critic verdict:** 5 × RETRY: missing specifics from aria_graph.py. Never reached APPROVED.

**Critic accurate?** ✅ Verdicts were correct — answer was always wrong. ⚠️ Round 3 anomaly: critic injected a hallucinated "Revised Answer" inline (fabricated a fake file review) rather than just issuing a RETRY directive.

**Retries:** 5 (all futile — planner never corrected tool choice)

**Filename attempted on first try:** ❌ Never — `read_file` not called at all

**Model identified correctly?** ❌ Never read the file

**MAX_ITERATIONS correct?** ❌ Not from file; Round 5 planner wrote "Based on finding 10 as the MAX_ITERATIONS value" — likely leaked from LinkedIn search result referencing this project

**Tool count correct?** ❌ web_search returned results about a different ARIA project ("50+ research ideas")

**Meta-awareness opinion reached?** ❌ Never

**Hallucinations spotted:**
- Call #1 Thought: *"I am ARIA… not equipped with code files or the ability to read local files"* — factually false; `read_file` was in the tool schema
- Calls #2–7: all web_search results described a biomedical ARIA project, not this agent
- Critic Round 3: fabricated a complete fake "Revised Answer" claiming to have reviewed `aria_graph.py`

**Failure mode:** Tool amnesia — model overrode the tool schema with its own prior belief that it couldn't read local files. Declared `read_file` unavailable at call #1, never reconsidered. The reflection loop (5 RETRYs) made this dramatically worse than M10 — more compute, more time, zero progress. Manually terminated at ~919s.

**Answer quality:** 0 / 3

**Time:** ~919s before manual stop (~15 min)

**vs M10 baseline (qwen2.5:3b best run):** M10 Run 1 hallucinated the filename (`RIA.py`), but at least tried `read_file`. v2 never attempted the tool at all — a complete regression in this failure mode. M10 best run: 2/3. v2: 0/3 DNF.  
**v2 fix applied:** `read_file` cap raised 3000→6000 chars — irrelevant this run since the tool was never called.

**Notes:**
- Root cause: small model's prior belief ("I can't read files") overpowered the tool schema. The system prompt lists tools by name but never explicitly says *"use read_file for local files"* — that instruction is needed for 3B models
- The LinkedIn post `ankit-chaudhary007 — Fixing Agent Loop with MAX_ITERATIONS in Python` appeared in 3 consecutive searches — likely a post about this project, creating a meta-loop where the agent searched for itself online
- Critic anomaly (Round 3): generating a fake answer inside the verdict is a new failure class — critic hallucinating content rather than just evaluating it
- Fix needed: system prompt should name tools explicitly for common task types, e.g. *"to read a local file, use read_file"*
- Filler: N/A (never reached a final answer)

---

## Summary

| # | Question | Iters | Retries | Critic | Answer /3 | Time | vs M10 |
|---|---|---|---|---|---|---|---|
| 1 | Bitcoin Price + Calc | 4 | 1 | APPROVED (R2) | 2 | 448.9s | n/a (new test) |
| 2 | Deep Chain — Mistral | 6 (DNF) | 3 | 3× RETRY / DNF | 0 | ~1167s (stopped) | identical failure to M10, slower |
| 3 | EMI Calculation | 4 (crash) | 1 | RETRY R1 / crash R2 | 0 | ~255s | rate ✅ formula ❌ (new crash) |
| 4 | Self-referential | 7 (DNF) | 5 | 5× RETRY / DNF | 0 | ~919s (stopped) | complete regression vs M10 |

**Overall notes:**
