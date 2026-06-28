"""A/B benchmark: ARIA v3 crew (multi-agent, cited) vs Tavily /research one-shot deep research.

Runs BOTH on the same topic and appends a side-by-side entry to Blackbox_vs_crew.md. The crew stays
the primary artifact — Tavily is the benchmark, not a replacement. Uses a SEPARATE Tavily key
(TAVILY_API_KEY2) so the experiment's credit burn is isolated from the crew's web_search.

If TAVILY_API_KEY2 lacks /research access or quota, this degrades gracefully (prints a note and
still runs/logs the crew side).

Run:  uv run python m13_multi_agent/Tavily_Research_blackbox.py
"""
import os
import sys
import time
import datetime

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from tavily import TavilyClient

try:  # works when run as a module / from the project root
    from m13_multi_agent.aria_v3_multiagent import run_blog_crew, SOCIAL_DOMAINS
except ModuleNotFoundError:  # works when run directly as a script (sibling on sys.path[0])
    from aria_v3_multiagent import run_blog_crew, SOCIAL_DOMAINS

load_dotenv()

AB_REPORT = os.path.join(os.path.dirname(__file__), "Blackbox_vs_crew.md")

# Terminal status values get_research() may report (matched case-insensitively).
_DONE_STATES = {"completed", "done", "succeeded", "success", "finished"}
_FAIL_STATES = {"failed", "error", "errored", "cancelled", "canceled"}


def _extract_report(resp: dict) -> str:
    """Pull the synthesized report text out of a get_research() response defensively —
    the beta response shape isn't pinned, so try the likely keys before falling back."""
    for key in ("report", "output", "answer", "result", "content", "research"):
        val = resp.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, dict):  # e.g. {"output": {"report": "..."}}
            for k2 in ("report", "text", "content", "answer"):
                if isinstance(val.get(k2), str) and val[k2].strip():
                    return val[k2]
    return ""


def _count_sources(resp: dict) -> int:
    for key in ("sources", "citations", "results"):
        val = resp.get(key)
        if isinstance(val, list):
            return len(val)
    return 0


def run_tavily_research(topic: str, allow_social: bool = False,
                        poll_timeout: float = 420.0, poll_interval: float = 6.0) -> dict:
    """Create a Tavily /research task and poll to completion. Returns a result dict with at least
    {ok: bool}. On success: {ok, report, sources, raw}. On any failure: {ok: False, error}."""
    key = os.getenv("TAVILY_API_KEY2")
    if not key:
        return {"ok": False, "error": "TAVILY_API_KEY2 is not set in .env"}

    client = TavilyClient(api_key=key)
    kwargs = {
        "model": "auto",
        "citation_format": "numbered",
        "output_length": "standard",
    }
    if not allow_social:
        kwargs["exclude_domains"] = SOCIAL_DOMAINS

    print(f"\n[tavily /research] creating task | allow_social={allow_social}")
    try:
        created = client.research(input=topic, **kwargs)
    except Exception as e:
        # Auth (401), forbidden/usage-limit (403), endpoint-not-enabled/beta (404), bad request — all land here.
        # Report the actual error rather than guessing the cause.
        return {"ok": False, "error": f"research() request failed ({type(e).__name__}: {e})"}

    request_id = created.get("request_id") if isinstance(created, dict) else None
    if not request_id:
        return {"ok": False, "error": f"no request_id in research() response: {created!r}"}

    print(f"[tavily /research] task {request_id} created; polling (timeout {poll_timeout:.0f}s)...")
    deadline = time.perf_counter() + poll_timeout
    last_status = None
    while time.perf_counter() < deadline:
        try:
            resp = client.get_research(request_id)
        except Exception as e:
            return {"ok": False, "error": f"get_research() failed: {type(e).__name__}: {e}"}

        status = str(resp.get("status", "")).lower() if isinstance(resp, dict) else ""
        if status != last_status:
            print(f"  [tavily /research] status: {status or '(none)'}")
            last_status = status

        report = _extract_report(resp) if isinstance(resp, dict) else ""
        if status in _DONE_STATES or (report and status not in _FAIL_STATES):
            return {"ok": True, "report": report, "sources": _count_sources(resp), "raw": resp}
        if status in _FAIL_STATES:
            return {"ok": False, "error": f"research task ended in status '{status}': {resp!r}"}

        time.sleep(poll_interval)

    return {"ok": False, "error": f"research task did not complete within {poll_timeout:.0f}s "
                                  f"(last status: {last_status})"}


def _log_ab(topic: str, allow_social: bool,
            crew: dict, crew_dt: float, tav: dict, tav_dt: float) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crew_draft = crew.get("draft", "")
    crew_sources = len(crew.get("sources", []))

    lines = [
        f"\n## A/B Run — {ts}",
        f"**Topic:** {topic}",
        f"**allow_social:** {allow_social}",
        "",
        "| Side | Latency | Sources | Status |",
        "|------|---------|---------|--------|",
        f"| Crew (ARIA v3) | {crew_dt:.1f}s | {crew_sources} | ok |",
        f"| Tavily /research | {tav_dt:.1f}s | {tav.get('sources', 0) if tav.get('ok') else '—'} "
        f"| {'ok' if tav.get('ok') else 'unavailable'} |",
        "",
        "### Crew draft",
        "",
        crew_draft or "_(empty)_",
        "",
        "### Tavily /research output",
        "",
        (tav.get("report") or "_(empty)_") if tav.get("ok") else f"_Unavailable: {tav.get('error')}_",
        "",
        "---",
    ]
    with open(AB_REPORT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_ab(topic: str, allow_social: bool = False) -> dict:
    print("=" * 60)
    print(f"A/B: crew vs Tavily /research\nTopic: {topic}")
    print("=" * 60)

    t0 = time.perf_counter()
    crew = run_blog_crew(topic, allow_social=allow_social)
    crew_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    tav = run_tavily_research(topic, allow_social=allow_social)
    tav_dt = time.perf_counter() - t1

    if not tav.get("ok"):
        print(f"\n[tavily /research] UNAVAILABLE — {tav.get('error')}")
        print("[tavily /research] Check TAVILY_API_KEY2's Tavily plan/quota and that the /research "
              "endpoint is enabled for it. Logging crew side only.")

    _log_ab(topic, allow_social, crew, crew_dt, tav, tav_dt)
    print(f"\nLogged A/B comparison to {AB_REPORT}")
    print(f"  crew: {crew_dt:.1f}s | tavily: {tav_dt:.1f}s ({'ok' if tav.get('ok') else 'unavailable'})")
    return {"crew": crew, "tavily": tav}


if __name__ == "__main__":
    run_ab("The commercial real estate crisis since 2020: remote work, office vacancies, and the looming debt wall")
