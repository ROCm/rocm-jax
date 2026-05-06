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

import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_job import BUCKETS as FAILURE_BUCKETS, JobAnalysis


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

def _render_classification_section(lines: list[str], rr: dict) -> None:
    """Append the six-bucket classification to the markdown."""
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
    window  = rr.get("window_days", 7)

    lines.append("## 🚦 Classification")
    lines.append("")
    lines.append(f"- Prior nightly window: **{window}** days "
                 f"({prior_n} prior nightly run(s) on file).")
    lines.append(f"- Continuous-CI evidence: **{cont_n}** run(s) "
                 f"strictly after the latest nightly.")
    lines.append("")
    lines.append("| Bucket | Definition | Count |")
    lines.append("|---|---|---:|")
    lines.append(f"| 🛑 **cancelled / infra** | latest job for the cell "
                 f"produced no pytest signal (cancelled / timed-out / "
                 f"infra-failed) | {counts['cancelled_infra']} |")
    lines.append(f"| ⚠️ **flaky** | rerun-passed in latest job, OR "
                 f"mixed pass/fail prior history | {counts['flaky']} |")
    lines.append(f"| ♻️ **chronic** | failed today + same `(gpu, py)` "
                 f"passed in continuous CI | {counts['chronic']} |")
    lines.append(f"| 🚨 **regression** | failed today + passed in **all** "
                 f"prior nightlies in the window | {counts['regression']} |")
    lines.append(f"| 🔁 **known** | failed today + failed in **all** prior "
                 f"nights the cell ran | {counts['known']} |")
    lines.append(f"| 🆕 **newly-failed** | failed today + cell or test had "
                 f"no full prior history (and never failed) "
                 f"| {counts['newly_failed']} |")
    lines.append("")

    confirmed = set(map(tuple, rr.get("stage2_continuous_confirmed", [])))

    def _row_badge(nodeid: str, cell: str, bucket: str) -> str:
        if bucket in ("regression", "known") and (nodeid, cell) in confirmed:
            return f"`{nodeid}` `+continuous`"
        return f"`{nodeid}`"

    def _table(title: str, rows: list[tuple[str, str]],
               bucket: str) -> None:
        if not rows:
            return
        lines.append(f"### {title} ({len(rows)})")
        lines.append("")
        by_node: dict[str, list[str]] = defaultdict(list)
        for nodeid, cell in rows:
            by_node[nodeid].append(cell)
        lines.append("| nodeid | cells affected |")
        lines.append("|---|---|")
        for nodeid, cells in sorted(by_node.items(),
                                    key=lambda kv: -len(kv[1])):
            badge = _row_badge(nodeid, cells[0], bucket)
            lines.append(f"| {badge} | {', '.join(sorted(cells))} |")
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
    lines.append(f"- Window: {classification.get('window_days', 7)} prior nights "
                 f"(history is keyed on `(nodeid, gpu, py)`; "
                 f"ROCm tag ignored).")
    lines.append("")

    # ---- Headline classification ----
    _render_classification_section(lines, classification)

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
                summary = (f.summary[:120].replace("\n", " ")
                           if f.summary else "")
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
</style>
</head><body>
<h1>JAX nightly Pytest-ROCm triage — {date}</h1>
<p>
  Run <a href="{url}">{run_id}</a> · {head_sha} ·
  conclusion: <b>{conclusion}</b> ·
  jobs: {n_jobs} ({n_failed} failed, {n_passed} passed)
</p>
<p class="small">Window: {window} prior nights. History keyed on
<code>(nodeid, gpu, py)</code> · ROCm tag ignored. Continuous-CI runs
strictly after the latest nightly contribute Stage-2 evidence.</p>

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


def _diff_section_html(title: str, items: list[tuple[str, str]],
                       confirmed: set) -> str:
    if not items:
        return ""
    out: list[str] = []
    by_node: dict[str, list[str]] = defaultdict(list)
    for n, c in items:
        by_node[n].append(c)
    out.append(f"<h3>{title} ({len(items)})</h3>")
    out.append('<table><tr><th>nodeid</th><th>cells</th></tr>')
    for nodeid, cells in sorted(by_node.items(),
                                key=lambda kv: -len(kv[1])):
        badge = ('<span class="badge">+continuous</span>'
                 if (nodeid, cells[0]) in confirmed else "")
        out.append(f'<tr><td class="nodeid">{nodeid}{badge}</td>'
                   f'<td>{", ".join(sorted(cells))}</td></tr>')
    out.append("</table>")
    return "\n".join(out)


def _render_classification_html(rr: dict) -> str:
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
    window  = rr.get("window_days", 7)
    confirmed = set(map(tuple, rr.get("stage2_continuous_confirmed", [])))

    parts = ["<h2>🚦 Classification</h2>",
             "<p>",
             f"Prior nightly window: <b>{window}</b> days "
             f"({prior_n} prior run(s) on file). "
             f"Continuous-CI evidence: <b>{cont_n}</b> run(s) "
             f"strictly after the latest nightly.",
             "</p>",
             "<table>",
             "<tr><th>Bucket</th><th>Count</th></tr>",
             f"<tr><td>🛑 cancelled / infra</td><td>{counts['cancelled_infra']}</td></tr>",
             f"<tr><td>⚠️ flaky</td><td>{counts['flaky']}</td></tr>",
             f"<tr><td>♻️ chronic</td><td>{counts['chronic']}</td></tr>",
             f"<tr><td>🚨 regression</td><td>{counts['regression']}</td></tr>",
             f"<tr><td>🔁 known</td><td>{counts['known']}</td></tr>",
             f"<tr><td>🆕 newly-failed</td><td>{counts['newly_failed']}</td></tr>",
             "</table>"]

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

    parts.append(_diff_section_html("⚠️ flaky", rr.get("flaky", []),
                                    confirmed))
    parts.append(_diff_section_html("♻️ chronic (passes in continuous CI)",
                                    rr.get("chronic", []), confirmed))
    parts.append(_diff_section_html(
        "🚨 regression (passed in all prior nightlies)",
        rr.get("regression", []), confirmed))
    parts.append(_diff_section_html(
        "🔁 known (failed in all prior nights cell ran)",
        rr.get("known", []), confirmed))
    parts.append(_diff_section_html(
        "🆕 newly-failed (no/partial prior history, never failed)",
        rr.get("newly_failed", []), confirmed))
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
        per_job_html.append(f'<h3>{j.matrix_cell} '
                            f'<a class="small" href="{url}">job {j.job_id}'
                            f'</a></h3>')
        if j.infra_events:
            per_job_html.append("<p>" + "".join(
                f'<span class="pill">{ev}</span>'
                for ev in j.infra_events) + "</p>")
        if j.flaky_tests:
            per_job_html.append(
                f'<p class="small">{len(j.flaky_tests)} rerun-passed '
                f'(flaky) test(s)</p>')
        if j.failures:
            per_job_html.append('<table><tr><th>bucket</th><th>nodeid</th>'
                                '<th>summary</th></tr>')
            for f in j.failures[:50]:
                per_job_html.append(
                    f'<tr><td>{f.bucket}</td>'
                    f'<td class="nodeid">{f.nodeid}</td>'
                    f'<td>{(f.summary or "")[:200]}</td></tr>')
            per_job_html.append("</table>")
            if len(j.failures) > 50:
                per_job_html.append(
                    f'<p class="small">... and {len(j.failures)-50} '
                    f'more</p>')
        else:
            per_job_html.append("<p><em>no test failures parsed</em></p>")

    return _HTML_TMPL.format(
        date=run_meta["date"],
        url=run_meta.get("html_url", ""),
        run_id=run_meta["run_id"],
        head_sha=(run_meta.get("head_sha") or "")[:8],
        conclusion=run_meta.get("conclusion", "unknown"),
        n_jobs=len(jobs),
        n_failed=sum(1 for j in jobs if j.conclusion == "failure"),
        n_passed=sum(1 for j in jobs if j.conclusion == "success"),
        window=classification.get("window_days", 7),
        classification_block=_render_classification_html(classification),
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
