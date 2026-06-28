# ARIA Agent — Stress Test Results

## Test 2: Deep Chain (search → read_url → write_file)
**Question:** Find Mistral's most recently released model, go to their official announcement or blog post for benchmark numbers, and save a structured summary to mistral_research.md.

| Metric | llama3.2:3b Run 1 | llama3.2:3b Run 2 | qwen2.5:3b Run 1 |
|---|---|---|---|
| LLM call #1 latency | 74.8s | 16.9s (model warm) | 50.0s |
| LLM call #2 latency | — | 76.9s (longer context) | 29.1s |
| LLM call #3 latency | — | — | 100.4s |
| Total time | crashed | 96.8s | 182.7s |
| Total LLM calls | crashed | 2 | 3 |
| Used web_search before read_url? | No — bad URL directly | No — bad URL first, web_search as fallback | ✅ Yes — web_search first, correct ordering |
| Used read_url to go deeper? | No | No | ❌ No — stopped at search snippets |
| Wrote non-empty file? | No — empty string | Yes — garbage HTML placeholder | ❌ No — narrated intent ("Let's create a summary") but never called write_file |
| Self-corrected on tool error? | No — crashed | Partial | N/A — no errors this run |
| Final answer quality | None (crashed) | Poor — hallucinated "Maverick" (Meta's model); fake benchmarks; invented pricing | Fair — correctly identified Mistral Small 4, real benchmark numbers, but sourced from snippets not the actual page |

### Notes
- **Run 1 failure cause:** `read_url` received `"Mistral official blog posts for benchmark numbers"` (a query string, not a URL). `tavily.extract()` threw `BadRequestError` and crashed the program.
- **Fix applied before Run 2:** URL validation guard in `execute_tool` — non-URLs now return a recoverable error string to the model instead of raising an exception.
- **llama3.2 behaviour flags (Run 1):** Emitted raw JSON in the Thought field; skipped `web_search` entirely; wrote an empty file immediately.
- **llama3.2 behaviour flags (Run 2):** Tried a hallucinated URL first (not web_search); wrote an HTML placeholder to file *before* searching — wrong order; final answer confuses "Maverick" (Meta's model) with a Mistral release; benchmark numbers and pricing are fabricated. URL validation fix did work — model recovered to web_search instead of crashing.
- **Timing note (llama3.2):** Call #1 dropped from 74.8s → 16.9s because the model was already loaded in Ollama's memory (cold vs warm load). Call #2 took 76.9s — longer context = more tokens to process on CPU.
- **qwen2.5:3b Run 1 behaviour:** Tool ordering correct (web_search first ✅). Identified Mistral Small 4 — a real model, no hallucination. Did not use read_url to go deeper. Called write_file as text intent in the final answer instead of as an actual tool call — the loop exited on `finish_reason == "stop"` before the file was written. Slower overall: 182.7s vs 96.8s for llama3.2 run 2, largely due to a 100s call #3.
- **write_file miss cause:** Model narrated "Let's create a summary file" in prose instead of emitting a tool call. Likely hit the reasoning/output boundary — `max_tokens=1024` may have cut the response short before the tool call was issued.

---

## Test 3: Calibration under Uncertainty
**Question:** What is Anthropic's current valuation, and how does it compare to OpenAI's? How confident are you in these numbers and why?

| Metric | qwen2.5:3b Run 1 |
|---|---|
| LLM call #1 latency | 51.4s |
| LLM call #2 latency | 79.6s |
| Total time | 133.0s |
| Total LLM calls | 2 |
| Used web_search? | ✅ Yes |
| Used read_url to verify snippet numbers? | ❌ No — accepted search snippets at face value |
| Expressed uncertainty explicitly? | ✅ Yes — "70% confident", acknowledged no official figures |
| Produced filler / chatbot-ism? | ❌ Yes — "Would you like me to search further?" violates "No filler" principle |
| Answer quality | Mixed — correct instinct to hedge; wrong to quote $965B/Anthropic and $852B/OpenAI as near-fact without read_url verification |

### Notes
- Calibration is meaningfully better than llama3.2: explicitly said "70% confident", noted figures come from valuation rounds not official disclosures.
- Still never uses `read_url` — relies on snippet text alone across both tests. This is a consistent weak point, not a one-off.
- The closing question ("Would you like me to...") is chatbot filler the system prompt explicitly bans. Small model habit bleeding through.

## Test 4: Formula-driven Calculation with Live Data
**Question:** Find the current RBI repo rate. Then calculate the monthly EMI on a ₹15 lakh home loan over 20 years using EMI = P·r(1+r)^n / ((1+r)^n − 1).
**Correct answer:** Rate = 5.25%, EMI ≈ ₹10,107/month

| Metric | qwen2.5:3b Run 1 |
|---|---|
| LLM call #1 latency | 50.3s |
| LLM call #2 latency | 89.0s |
| LLM call #3 latency | 32.7s |
| Total time | 175.8s |
| Total LLM calls | 3 |
| Used web_search to find repo rate? | ❌ No — went straight to hallucinated read_url |
| read_url prompt rule took effect? | Partial — model tried read_url but hallucinated the URL (`rbi.org.in/Scripts/命中的.aspx` — contains Chinese chars) |
| Repo rate found | ❌ 4.75% (actual: 5.25%) |
| Used correct EMI formula? | ❌ No — used `150000 * 4.75 / 96`, a meaningless expression |
| Principal used | ❌ ₹1,50,000 (10× too small; should be ₹15,00,000) |
| EMI answer | ❌ ₹7,421.88 (correct: ~₹10,107) |
| write_file path valid? | ❌ Linux-style path `/user/...` — fails on Windows |
| Filler phrases? | ❌ Yes — "If you need further assistance, please let me know" |

### Notes
- **Three compounding failures:** wrong rate source, wrong principal (off by 10×), wrong formula. None of the three inputs to the EMI calculation were correct.
- **read_url prompt rule backfired:** the model now tries read_url but invents a URL instead of searching first to find one. The rule needs to explicitly state `web_search first to get a URL, then read_url` — not just "follow up with read_url."
- **Windows path bug:** model hardcoded a Unix path. Could add a note to the system prompt, or just accept it as a small-model limitation.
- **Filler persists** despite "No filler" instruction.

## Test 5: Self-referential / read_file
**Question:** Read ARIA.py and tell me: model in use, number of tools, MAX_ITERATIONS cap. Then give your honest opinion on whether 10 iterations is enough.
**Ground truth:** qwen2.5:3b, 5 tools, MAX_ITERATIONS = 10

| Metric | qwen2.5:3b Run 1 |
|---|---|
| LLM call #1 latency | 14.6s |
| LLM call #2 latency | 15.0s |
| LLM call #3 latency | 18.9s |
| Total time | 48.5s (fastest run — no web calls) |
| Total LLM calls | 3 |
| Used correct tool (read_file)? | ✅ Yes |
| Correct filename on first try? | ❌ Tried `RIA.py` — dropped the `A` from `ARIA.py` |
| Self-corrected filename? | ❌ Retried same wrong filename twice, then gave up |
| Final answer | ❌ None — asked user to verify the filename |
| Meta-awareness opinion reached? | ❌ Never got there |

| | **Run 1** | **Run 2** | **Run 3** |
|---|---|---|---|
| Filename attempted | `RIA.py` ❌ | `ARIA.py` ✅ | `ARIA.py` ✅ |
| File read succeeded | ❌ | ✅ | ✅ |
| Called read_file twice? | No | No | ❌ Yes — redundant duplicate |
| LLM call #1 latency | 14.6s | n/a | 129.6s |
| LLM call #2 latency | 15.0s | timed out | 186.8s |
| Total time | 48.5s | DNF | 316.6s |
| Model identified correctly? | ❌ | n/a | ✅ `qwen2.5:3b` |
| MAX_ITERATIONS correct? | ❌ | n/a | ✅ 10 |
| Tools count correct? | ❌ | n/a | ❌ Said 3 — truncation cut off `read_file` and `write_file` |
| Meta-awareness opinion? | ❌ | n/a | ✅ Argued 10 is insufficient for multi-hop — reasonable and self-aware |

### Notes
- Run 1: one-letter hallucination (`RIA` vs `ARIA`), retried same wrong name twice — zero self-correction.
- Run 2: filename correct, file read succeeded — but full file flooded context; timed out on CPU.
- Run 3: truncation fix prevented timeout, but introduced a new error — 3000-char cap cuts off before `read_file` and `write_file` definitions, so model reported 3 tools instead of 5. The fix that solved the crash created a factual blind spot.
- **Best result yet:** correct model, correct MAX_ITERATIONS, thoughtful meta-opinion on iteration limits.
- **Filler persists:** "Therefore, I would recommend more iterations if necessary." — still bleeds through despite system prompt ban.
