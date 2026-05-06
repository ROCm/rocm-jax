"""Render markdown / HTML / JSON triage reports.

The renderers are deliberately self-contained -- no jinja2 dependency, just
f-strings + a tiny HTML template -- so a fresh checkout runs without any
``pip install``.

The single classification source of truth is the ``classification`` dict
returned by ``regression.regression_classify``.  Six buckets:

    cancelled_infra | flaky | chronic | regression | known | newly-failed

Stage-2 confirmation (continuous-CI also failed the test) is rendered as a
``+continuous`` badge on rows in ``regression`` and ``known``.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analyze_job import BUCKETS as FAILURE_BUCKETS, Failure, JobAnalysis


# Maximum characters of a failure summary to render inline in the headline
# classification tables. Longer messages are truncated with a U+2026 marker.
# Per-job tables (further down the report) always show the full first 200
# chars; the headline tables prefer a tighter excerpt so the row stays
# readable at table width.
SUMMARY_INLINE_MAX = 160

# Per-job table summary cap (full text, not truncated to inline width).
SUMMARY_PER_JOB_MAX = 400

# Pytest renders the actual exception message inside the FAILURES section as
# lines that start with ``E   `` (one ``E``, then whitespace, then content).
# Empty ``E`` spacer lines are skipped.
_E_LINE = re.compile(r"^E\s+(.*\S)\s*$")

# A pytest short-summary line that boils down to just an exception class
# name -- ``AssertionError:``, ``RuntimeError`` -- is uninformative because
# it tells you nothing about *why* the test failed.  When we see one of
# these we prefer building the displayed summary from the excerpt's
# ``E   ...`` lines instead.  Recognises common Python exception suffixes
# (``Error``, ``Exception``, ``Failure``, ``Warning``).
_BARE_EXC_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure|Warning)\s*:?\s*$")


def _e_lines_text(excerpt: str) -> str:
    """Join the ``E   ...`` content lines of a pytest excerpt into a single
    string, preserving order.  Returns ``""`` if the excerpt has none.
    """
    keep: list[str] = []
    for line in (excerpt or "").splitlines():
        m = _E_LINE.match(line.strip())
        if m:
            keep.append(m.group(1).strip())
    return "  ".join(keep)


def _failure_display_text(f: Failure) -> str:
    """Pick the best human-readable text to show for a single failure.

    Falls back to the traceback excerpt's ``E   ...`` lines when the
    pytest short-summary line is empty or is just a bare exception class
    name with no message (e.g. ``AssertionError:``).  This is what makes
    rows like ``testApplyAlongAxis5`` actually say *why* they failed
    instead of just echoing ``AssertionError:``.
    """
    summary = (f.summary or "").strip()
    excerpt = (f.excerpt or "").strip()
    if summary and not _BARE_EXC_RE.match(summary):
        return summary
    enriched = _e_lines_text(excerpt)
    if enriched:
        return enriched
    return summary or excerpt


def _build_failure_summaries(
        jobs: list[JobAnalysis]) -> dict[tuple[str, str], str]:
    """Return ``(nodeid, matrix_cell) -> excerpt`` for the headline tables.

    Uses :func:`_failure_display_text` so a row whose short-summary is just
    ``AssertionError:`` still gets a useful message spliced in from the
    traceback excerpt.  The first non-empty match per
    ``(nodeid, matrix_cell)`` wins.
    """
    out: dict[tuple[str, str], str] = {}
    for j in jobs:
        for f in j.failures:
            key = (f.nodeid, j.matrix_cell)
            if key in out:
                continue
            text = _failure_display_text(f)
            if text:
                out[key] = text
    return out


def _excerpt_for_row(nodeid: str, cells: list[str],
                     summaries: dict[tuple[str, str], str] | None) -> str:
    """Pick a representative summary for a (nodeid, cells) row.

    Returns "" if no excerpt is available (so callers can render an em-dash
    or omit the column gracefully).
    """
    if not summaries:
        return ""
    for c in cells:
        text = summaries.get((nodeid, c))
        if text:
            text = " ".join(text.split())  # collapse whitespace / newlines
            if len(text) > SUMMARY_INLINE_MAX:
                text = text[:SUMMARY_INLINE_MAX - 1].rstrip() + "\u2026"
            return text
    return ""


# Headline buckets in priority order (this is the order the report uses).
HEADLINE_BUCKETS = (
    "cancelled_infra",
    "flaky",
    "chronic",
    "regression",
    "known",
    "newly_failed",
)

_BUCKET_LABELS = {
    "cancelled_infra": "🛑 cancelled / infra",
    "flaky":           "⚠️ flaky",
    "chronic":         "♻️ chronic (passes in continuous)",
    "regression":      "🚨 regression",
    "known":           "🔁 known",
    "newly_failed":      "🆕 newly-failed",
}

# Single source of truth for the per-bucket definitions rendered in the
# headline classification table.  Kept short enough to fit a table cell
# without wrapping more than a couple of lines; deeper detail belongs in
# the README, not the daily report.
_BUCKET_DEFINITIONS = {
    "cancelled_infra":
        "Latest job for the cell produced no pytest signal — cancelled, "
        "timed out, or infra-failed before tests ran. Not a code-level "
        "failure; retry the runner.",
    "flaky":
        "Test rerun-passed in the latest job, OR has a mixed pass/fail "
        "history across prior nightlies. Non-deterministic; not necessarily "
        "a regression.",
    "chronic":
        "Failed today AND the same (gpu, py) cell currently passes in "
        "continuous CI runs that ran after this nightly. Likely a stale "
        "nightly artefact, not a real regression.",
    "regression":
        "Failed today AND passed in every prior nightly inside the window. "
        "Most actionable bucket — points at a recent code change.",
    "known":
        "Failed today AND failed in every prior nightly the cell ran. "
        "Pre-existing failure, not a new regression; usually waiting on "
        "a fix.",
    "newly_failed":
        "Failed today with no full prior history (test or cell is new, or "
        "the window is not yet warm) and was never previously seen "
        "failing. Investigate as a possible regression once history "
        "accumulates.",
}


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def render_json(*, run_meta: dict, jobs: list[JobAnalysis],
                classification: dict) -> str:
    payload: dict = {
        "run":   run_meta,
        "jobs":  [j.to_dict() for j in jobs],
        "classification": _json_safe_classification(classification),
    }
    return json.dumps(payload, indent=2, default=str)


def _json_safe_classification(c: dict) -> dict:
    """Convert tuple-of-tuples to list-of-lists for JSON friendliness."""
    out = {**c}
    for k in ("regression", "known", "chronic", "flaky", "newly_failed",
              "stage2_continuous_confirmed"):
        if k in out:
            out[k] = [list(t) for t in out[k]]
    if "cancelled_infra" in out:
        out["cancelled_infra"] = [
            {"matrix_cell": cell, "reason": reason, "events": list(events)}
            for cell, reason, events in out["cancelled_infra"]
        ]
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _render_classification_section(
        lines: list[str], rr: dict,
        summaries: dict[tuple[str, str], str] | None = None) -> None:
    """Append the six-bucket classification to the markdown.

    If ``summaries`` is provided (a ``(nodeid, matrix_cell) -> excerpt``
    map, typically built via ``_build_failure_summaries(jobs)``), the
    per-test bucket tables grow a third ``summary`` column so the
    headline view shows *why* each test failed without scrolling down to
    the per-job section.
    """
    counts = {
        "cancelled_infra": len(rr.get("cancelled_infra", [])),
        "flaky":           len(rr.get("flaky", [])),
        "chronic":         len(rr.get("chronic", [])),
        "regression":      len(rr.get("regression", [])),
        "known":           len(rr.get("known", [])),
        "newly_failed":      len(rr.get("newly_failed", [])),
    }
    prior_n = len(rr.get("prior_nightly_run_ids") or [])
    cont_n  = len(rr.get("continuous_runs_used") or [])
    window  = rr.get("window_days", 6)

    lines.append("## 🚦 Classification")
    lines.append("")
    lines.append(f"- Prior-nightly window configured: **{window}** day(s).")
    lines.append(f"- Prior nightlies actually used in this analysis: "
                 f"**{prior_n}** of up to {window} (missing nights inside "
                 f"the window are silently ignored).")
    lines.append(f"- Continuous-CI evidence: **{cont_n}** run(s) "
                 f"strictly after the latest nightly.")
    lines.append("")
    def _md_def(text: str) -> str:
        # Pipe is the column separator in markdown tables -- escape any
        # literal pipes that might sneak into a definition.
        return text.replace("|", "\\|")

    lines.append("| Bucket | Definition | Count |")
    lines.append("|---|---|---:|")
    for b in HEADLINE_BUCKETS:
        lines.append(
            f"| **{_BUCKET_LABELS[b]}** | {_md_def(_BUCKET_DEFINITIONS[b])} "
            f"| {counts[b]} |")
    lines.append("")

    confirmed = set(map(tuple, rr.get("stage2_continuous_confirmed", [])))

    def _row_badge(nodeid: str, cell: str, bucket: str) -> str:
        if bucket in ("regression", "known") and (nodeid, cell) in confirmed:
            return f"`{nodeid}` `+continuous`"
        return f"`{nodeid}`"

    def _md_summary(text: str) -> str:
        """Escape ``|`` (column separator) and collapse newlines for an
        inline markdown table cell."""
        if not text:
            return "—"
        return text.replace("\\", "\\\\").replace("|", "\\|")

    def _table(title: str, rows: list[tuple[str, str]],
               bucket: str) -> None:
        if not rows:
            return
        lines.append(f"### {title} ({len(rows)})")
        lines.append("")
        by_node: dict[str, list[str]] = defaultdict(list)
        for nodeid, cell in rows:
            by_node[nodeid].append(cell)
        if summaries is not None:
            lines.append("| nodeid | cells affected | summary |")
            lines.append("|---|---|---|")
        else:
            lines.append("| nodeid | cells affected |")
            lines.append("|---|---|")
        for nodeid, cells in sorted(by_node.items(),
                                    key=lambda kv: -len(kv[1])):
            badge = _row_badge(nodeid, cells[0], bucket)
            cells_sorted = sorted(cells)
            cells_md = ", ".join(cells_sorted)
            if summaries is not None:
                excerpt = _excerpt_for_row(nodeid, cells_sorted, summaries)
                lines.append(f"| {badge} | {cells_md} "
                             f"| {_md_summary(excerpt)} |")
            else:
                lines.append(f"| {badge} | {cells_md} |")
        lines.append("")

    # Render in priority order so the most-actionable buckets come first.
    if rr.get("cancelled_infra"):
        lines.append(f"### 🛑 cancelled / infra ({len(rr['cancelled_infra'])})")
        lines.append("")
        lines.append("| matrix cell | reason | events |")
        lines.append("|---|---|---|")
        for cell, reason, events in rr["cancelled_infra"]:
            ev = ", ".join(events) if events else "—"
            lines.append(f"| `{cell}` | `{reason}` | {ev} |")
        lines.append("")

    _table("⚠️ flaky", rr.get("flaky", []),         "flaky")
    _table("♻️ chronic (passes in continuous CI)",
           rr.get("chronic", []),                   "chronic")
    _table("🚨 regression (passed in all prior nightlies)",
           rr.get("regression", []),                "regression")
    _table("🔁 known (failed in all prior nights cell ran)",
           rr.get("known", []),                     "known")
    _table("🆕 newly-failed (no/partial prior history, never failed)",
           rr.get("newly_failed", []),                "newly_failed")


def _bucket_counter(jobs: list[JobAnalysis]) -> Counter:
    c = Counter()
    for j in jobs:
        for f in j.failures:
            c[f.bucket] += 1
    return c


def _matrix_index(jobs: list[JobAnalysis]
                  ) -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
    """Build a dense (gpu_config x py) matrix of failure counts.

    The keys come from ``matrix_cell = "<gpu>-py<py>-rocm<rocm>"``.
    """
    rows = sorted({j.matrix_cell.split("-")[0] for j in jobs})
    cols = sorted({j.matrix_cell.split("-")[1] for j in jobs
                   if "-" in j.matrix_cell})
    grid: dict[tuple[str, str], int] = {}
    for j in jobs:
        parts = j.matrix_cell.split("-")
        if len(parts) < 2:
            continue
        gpu, py = parts[0], parts[1]
        grid[(gpu, py)] = grid.get((gpu, py), 0) + len(j.failures)
    return rows, cols, grid


def render_markdown(*, run_meta: dict, jobs: list[JobAnalysis],
                    classification: dict) -> str:
    lines: list[str] = []
    head_sha = (run_meta.get("head_sha") or "")[:8]
    lines.append(f"# JAX nightly Pytest-ROCm triage — {run_meta['date']}")
    lines.append("")
    lines.append(f"- Run: [{run_meta['run_id']}]({run_meta.get('html_url','')}) "
                 f"({run_meta.get('conclusion', 'unknown')})")
    if head_sha:
        lines.append(f"- HEAD: `{head_sha}`")
    lines.append(f"- Jobs analyzed: **{len(jobs)}** "
                 f"({sum(1 for j in jobs if j.conclusion=='failure')} failed, "
                 f"{sum(1 for j in jobs if j.conclusion=='success')} passed, "
                 f"{sum(1 for j in jobs if j.conclusion not in ('failure','success'))} other)")
    lines.append(f"- Window: up to {classification.get('window_days', 6)} "
                 f"prior nights considered "
                 f"(history is keyed on `(nodeid, gpu, py)`; "
                 f"ROCm tag ignored).")
    lines.append("")

    # ---- Headline classification ----
    summaries = _build_failure_summaries(jobs)
    _render_classification_section(lines, classification, summaries)

    # ---- Failure-bucket distribution (per-failure infra/build/test bucket) ----
    counter = _bucket_counter(jobs)
    if counter:
        lines.append("## Per-failure category")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for b in FAILURE_BUCKETS:
            n = counter.get(b, 0)
            if n:
                lines.append(f"| {b} | {n} |")
        lines.append("")

    # ---- Cell matrix heatmap ----
    rows_, cols_, grid = _matrix_index(jobs)
    if rows_ and cols_:
        lines.append("## Failure-count matrix (rows = GPU, cols = Python)")
        lines.append("")
        header = "| | " + " | ".join(cols_) + " |"
        sep    = "|---|" + "|".join(["---:"] * len(cols_)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in rows_:
            cells = [str(grid.get((r, c), 0)) or "·" for c in cols_]
            lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- Per-job summary ----
    lines.append("## Per-job summary")
    lines.append("")
    for j in sorted(jobs, key=lambda x: x.matrix_cell):
        url = (f"https://github.com/{run_meta.get('repo','jax-ml/jax')}"
               f"/actions/runs/{run_meta['run_id']}/job/{j.job_id}")
        status = ("❌" if j.conclusion == "failure"
                  else ("✅" if j.conclusion == "success" else "⚠️"))
        lines.append(f"### {status} `{j.matrix_cell}` — "
                     f"[{j.job_id}]({url})  ({j.duration_s}s)")
        if j.infra_events:
            lines.append(f"- infra events: {', '.join(j.infra_events)}")
        if j.flaky_tests:
            lines.append(f"- {len(j.flaky_tests)} rerun-passed (flaky) test(s)")
        if not j.failures:
            lines.append("- no test failures parsed")
        else:
            lines.append(f"- {len(j.failures)} failures:")
            for f in j.failures[:10]:
                text = _failure_display_text(f)
                summary = " ".join(text.split())[:SUMMARY_PER_JOB_MAX]
                lines.append(f"  - `[{f.bucket}]` `{f.nodeid}` — {summary}")
            if len(j.failures) > 10:
                lines.append(f"  - ... and {len(j.failures) - 10} more")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_TMPL = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>JAX nightly ROCm triage — {date}</title>
<style>
 body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
        margin: 24px; color: #1f2328; }}
 h1 {{ margin-top: 0; }}
 table {{ border-collapse: collapse; margin-bottom: 24px; }}
 td, th {{ border: 1px solid #d0d7de; padding: 6px 10px; font-size: 13px; }}
 th {{ background: #f6f8fa; }}
 .heat0 {{ background: #d9f7d9; }}
 .heat1 {{ background: #fff3b0; }}
 .heat2 {{ background: #ffd0a3; }}
 .heat3 {{ background: #ffb3b3; }}
 .heat4 {{ background: #ff7b7b; color: white; }}
 .nodeid {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
 .small {{ color: #57606a; font-size: 12px; }}
 .pill {{ display: inline-block; padding: 1px 8px; border-radius: 10px;
         background: #eaeef2; font-size: 11px; margin-right: 4px; }}
 .badge {{ display: inline-block; padding: 1px 6px; border-radius: 6px;
          background: #ffe5e0; color: #b1300a; font-size: 11px;
          margin-left: 6px; }}
 .excerpt {{ display: block; max-width: 720px; color: #57606a;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px; word-break: break-word;
            white-space: normal; }}
 table.bucket-legend td.bucket {{ white-space: nowrap; font-weight: 600; }}
 table.bucket-legend td.bucket-def {{ max-width: 640px; color: #57606a;
            font-size: 12px; line-height: 1.4; }}
 table.bucket-legend td.num,
 table.bucket-legend th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head><body>
<h1>JAX nightly Pytest-ROCm triage — {date}</h1>
<p>
  Run <a href="{url}">{run_id}</a> · {head_sha} ·
  conclusion: <b>{conclusion}</b> ·
  jobs: {n_jobs} ({n_failed} failed, {n_passed} passed)
</p>
<p class="small">Window: up to {window} prior nights considered.
History keyed on <code>(nodeid, gpu, py)</code> · ROCm tag ignored.
Continuous-CI runs strictly after the latest nightly contribute
Stage-2 evidence.</p>

{classification_block}

<h2>Failure-count matrix</h2>
{matrix_table}

<h2>Per-failure category</h2>
{bucket_table}

<h2>Per-job</h2>
{per_job}

</body></html>
"""


def _heat_class(n: int) -> str:
    if n == 0:    return "heat0"
    if n <= 5:    return "heat1"
    if n <= 20:   return "heat2"
    if n <= 100:  return "heat3"
    return "heat4"


def _diff_section_html(
        title: str, items: list[tuple[str, str]],
        confirmed: set,
        summaries: dict[tuple[str, str], str] | None = None) -> str:
    if not items:
        return ""
    out: list[str] = []
    by_node: dict[str, list[str]] = defaultdict(list)
    for n, c in items:
        by_node[n].append(c)
    out.append(f"<h3>{html.escape(title)} ({len(items)})</h3>")
    if summaries is not None:
        out.append('<table><tr><th>nodeid</th><th>cells</th>'
                   '<th>summary</th></tr>')
    else:
        out.append('<table><tr><th>nodeid</th><th>cells</th></tr>')
    for nodeid, cells in sorted(by_node.items(),
                                key=lambda kv: -len(kv[1])):
        cells_sorted = sorted(cells)
        badge = ('<span class="badge">+continuous</span>'
                 if (nodeid, cells_sorted[0]) in confirmed else "")
        nodeid_html = f'{html.escape(nodeid)}{badge}'
        cells_html = html.escape(", ".join(cells_sorted))
        if summaries is not None:
            excerpt = _excerpt_for_row(nodeid, cells_sorted, summaries)
            excerpt_html = (f'<span class="excerpt">{html.escape(excerpt)}'
                            f'</span>' if excerpt
                            else '<span class="small">—</span>')
            out.append(f'<tr><td class="nodeid">{nodeid_html}</td>'
                       f'<td>{cells_html}</td>'
                       f'<td>{excerpt_html}</td></tr>')
        else:
            out.append(f'<tr><td class="nodeid">{nodeid_html}</td>'
                       f'<td>{cells_html}</td></tr>')
    out.append("</table>")
    return "\n".join(out)


def _render_classification_html(
        rr: dict,
        summaries: dict[tuple[str, str], str] | None = None) -> str:
    counts = {
        "cancelled_infra": len(rr.get("cancelled_infra", [])),
        "flaky":           len(rr.get("flaky", [])),
        "chronic":         len(rr.get("chronic", [])),
        "regression":      len(rr.get("regression", [])),
        "known":           len(rr.get("known", [])),
        "newly_failed":      len(rr.get("newly_failed", [])),
    }
    prior_n = len(rr.get("prior_nightly_run_ids") or [])
    cont_n  = len(rr.get("continuous_runs_used") or [])
    window  = rr.get("window_days", 6)
    confirmed = set(map(tuple, rr.get("stage2_continuous_confirmed", [])))

    parts = ["<h2>🚦 Classification</h2>",
             "<p>",
             f"Prior-nightly window configured: <b>{window}</b> day(s). "
             f"Prior nightlies actually used in this analysis: "
             f"<b>{prior_n}</b> of up to {window} (missing nights inside "
             f"the window are silently ignored). "
             f"Continuous-CI evidence: <b>{cont_n}</b> run(s) "
             f"strictly after the latest nightly.",
             "</p>",
             '<table class="bucket-legend">',
             "<tr><th>Bucket</th><th>Definition</th>"
             '<th class="num">Count</th></tr>']
    for b in HEADLINE_BUCKETS:
        parts.append(
            f'<tr><td class="bucket">{html.escape(_BUCKET_LABELS[b])}</td>'
            f'<td class="bucket-def">{html.escape(_BUCKET_DEFINITIONS[b])}'
            f'</td>'
            f'<td class="num">{counts[b]}</td></tr>')
    parts.append("</table>")

    if rr.get("cancelled_infra"):
        parts.append(
            f"<h3>🛑 cancelled / infra ({counts['cancelled_infra']})</h3>")
        parts.append("<table><tr><th>matrix cell</th><th>reason</th>"
                     "<th>events</th></tr>")
        for cell, reason, events in rr["cancelled_infra"]:
            ev = ", ".join(events) if events else "—"
            parts.append(f'<tr><td class="nodeid">{cell}</td>'
                         f'<td>{reason}</td><td>{ev}</td></tr>')
        parts.append("</table>")

    parts.append(_diff_section_html(
        "⚠️ flaky", rr.get("flaky", []), confirmed, summaries))
    parts.append(_diff_section_html(
        "♻️ chronic (passes in continuous CI)",
        rr.get("chronic", []), confirmed, summaries))
    parts.append(_diff_section_html(
        "🚨 regression (passed in all prior nightlies)",
        rr.get("regression", []), confirmed, summaries))
    parts.append(_diff_section_html(
        "🔁 known (failed in all prior nights cell ran)",
        rr.get("known", []), confirmed, summaries))
    parts.append(_diff_section_html(
        "🆕 newly-failed (no/partial prior history, never failed)",
        rr.get("newly_failed", []), confirmed, summaries))
    return "\n".join(filter(None, parts))


def render_html(*, run_meta: dict, jobs: list[JobAnalysis],
                classification: dict) -> str:
    rows_, cols_, grid = _matrix_index(jobs)
    matrix_html = ['<table><tr><th></th>']
    matrix_html += [f"<th>{c}</th>" for c in cols_]
    matrix_html.append('</tr>')
    for r in rows_:
        matrix_html.append(f"<tr><th>{r}</th>")
        for c in cols_:
            n = grid.get((r, c), 0)
            matrix_html.append(f'<td class="{_heat_class(n)}">{n}</td>')
        matrix_html.append("</tr>")
    matrix_html.append("</table>")

    counter = _bucket_counter(jobs)
    bucket_html = ['<table><tr><th>Category</th><th>Count</th></tr>']
    for b in FAILURE_BUCKETS:
        if counter.get(b, 0):
            bucket_html.append(f"<tr><td>{b}</td><td>{counter[b]}</td></tr>")
    bucket_html.append("</table>")

    per_job_html: list[str] = []
    for j in sorted(jobs, key=lambda x: x.matrix_cell):
        url = (f"https://github.com/{run_meta.get('repo','jax-ml/jax')}"
               f"/actions/runs/{run_meta['run_id']}/job/{j.job_id}")
        per_job_html.append(f'<h3>{html.escape(j.matrix_cell)} '
                            f'<a class="small" href="{html.escape(url)}">'
                            f'job {j.job_id}</a></h3>')
        if j.infra_events:
            per_job_html.append("<p>" + "".join(
                f'<span class="pill">{html.escape(ev)}</span>'
                for ev in j.infra_events) + "</p>")
        if j.flaky_tests:
            per_job_html.append(
                f'<p class="small">{len(j.flaky_tests)} rerun-passed '
                f'(flaky) test(s)</p>')
        if j.failures:
            per_job_html.append('<table><tr><th>bucket</th><th>nodeid</th>'
                                '<th>summary</th></tr>')
            for f in j.failures[:50]:
                text = _failure_display_text(f)
                summary_text = " ".join(text.split())[:SUMMARY_PER_JOB_MAX]
                per_job_html.append(
                    f'<tr><td>{html.escape(f.bucket)}</td>'
                    f'<td class="nodeid">{html.escape(f.nodeid)}</td>'
                    f'<td class="excerpt">{html.escape(summary_text)}</td></tr>')
            per_job_html.append("</table>")
            if len(j.failures) > 50:
                per_job_html.append(
                    f'<p class="small">... and {len(j.failures)-50} '
                    f'more</p>')
        else:
            per_job_html.append("<p><em>no test failures parsed</em></p>")

    summaries = _build_failure_summaries(jobs)
    return _HTML_TMPL.format(
        date=run_meta["date"],
        url=run_meta.get("html_url", ""),
        run_id=run_meta["run_id"],
        head_sha=(run_meta.get("head_sha") or "")[:8],
        conclusion=run_meta.get("conclusion", "unknown"),
        n_jobs=len(jobs),
        n_failed=sum(1 for j in jobs if j.conclusion == "failure"),
        n_passed=sum(1 for j in jobs if j.conclusion == "success"),
        window=classification.get("window_days", 6),
        classification_block=_render_classification_html(
            classification, summaries),
        matrix_table="\n".join(matrix_html),
        bucket_table="\n".join(bucket_html),
        per_job="\n".join(per_job_html),
    )


# ---------------------------------------------------------------------------
# Convenience: write all three side by side
# ---------------------------------------------------------------------------

def write_all(out_dir: Path, *, run_meta: dict,
              jobs: list[JobAnalysis],
              classification: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json":     out_dir / "summary.json",
        "markdown": out_dir / "report.md",
        "html":     out_dir / "report.html",
    }
    # Always write UTF-8: report.md/report.html contain emoji that Python's
    # locale-default codec on Windows (cp1252) cannot encode.
    paths["json"].write_text(render_json(
        run_meta=run_meta, jobs=jobs,
        classification=classification), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(
        run_meta=run_meta, jobs=jobs,
        classification=classification), encoding="utf-8")
    paths["html"].write_text(render_html(
        run_meta=run_meta, jobs=jobs,
        classification=classification), encoding="utf-8")
    return paths
