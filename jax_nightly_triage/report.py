"""Render markdown / HTML / JSON triage reports.

The renderers are deliberately self-contained -- no jinja2 dependency, just
f-strings + a tiny HTML template -- so a fresh checkout runs without any
``pip install``.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from analyze_job import BUCKETS, JobAnalysis


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def render_json(*, run_meta: dict, jobs: list[JobAnalysis],
                diff: dict, regression_result: dict | None = None) -> str:
    payload: dict = {
        "run":   run_meta,
        "jobs":  [j.to_dict() for j in jobs],
        "diff":  {
            **diff,
            "new":       [list(t) for t in diff["new"]],
            "recurring": [list(t) for t in diff["recurring"]],
            "flaky":     [list(t) for t in diff["flaky"]],
            "recovered": [list(t) for t in diff["recovered"]],
            "prior_nights_seen": [
                {"nodeid": k[0], "matrix_cell": k[1], "nights_failed": v}
                for k, v in diff["prior_nights_seen"].items()
            ],
        },
    }
    if regression_result is not None:
        payload["regression"] = {
            **regression_result,
            "regression":      [list(t) for t in regression_result["regression"]],
            "chronic":         [list(t) for t in regression_result["chronic"]],
            "chronic_pending": [list(t) for t in regression_result["chronic_pending"]],
            "newly_broken":    [list(t) for t in regression_result["newly_broken"]],
            "known":           [list(t) for t in regression_result["known"]],
            "new":             [list(t) for t in regression_result["new"]],
        }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _render_regression_section(lines: list[str], rr: dict) -> None:
    """Append the regression classification to the markdown."""
    n_reg     = len(rr["regression"])
    n_chronic = len(rr["chronic"])
    n_cpend   = len(rr["chronic_pending"])
    n_newly   = len(rr["newly_broken"])
    n_known   = len(rr["known"])
    n_new     = len(rr["new"])
    prior     = rr.get("prior_nightly_run_id")
    cont_n    = len(rr.get("continuous_runs_used") or [])
    window    = rr.get("window_days", 7)
    threshold = rr.get("chronic_threshold", 4)

    lines.append("## 🚨 Regression check "
                 f"(chronic ≥{threshold}-of-{window} + continuous CI)")
    lines.append("")
    if prior is None:
        lines.append("- No prior nightly run in the database -- "
                     "history-based buckets are unavailable.  Today's "
                     "failures are labelled `NEW` until a second nightly "
                     "is ingested.")
    else:
        lines.append(f"- Immediately prior nightly: `{prior}` "
                     f"(used for KNOWN vs NEWLY_BROKEN).")
    lines.append(f"- Chronic window: {window} prior nights, "
                 f"threshold ≥{threshold} of {window}.")
    lines.append(f"- Continuous-CI evidence: {cont_n} run(s) within the "
                 f"window or after this nightly.")
    lines.append("")
    lines.append("| Bucket | Definition | Count |")
    lines.append("|---|---|---:|")
    lines.append(f"| 🚨 **REGRESSION** | failed today + chronic in nightly "
                 f"(≥{threshold}-of-{window}) + failed in continuous CI "
                 f"| {n_reg} |")
    lines.append(f"| ♻️ **CHRONIC** | failed today + chronic in nightly, "
                 f"continuous CI shows it passing (likely env-specific or "
                 f"fixed in HEAD) | {n_chronic} |")
    lines.append(f"| 🕒 **CHRONIC_PENDING** | failed today + chronic in "
                 f"nightly, but no continuous-CI evidence yet | {n_cpend} |")
    lines.append(f"| 🆕 **NEWLY_BROKEN** | failed today, passed in the "
                 f"immediately prior nightly, not yet chronic | {n_newly} |")
    lines.append(f"| ♻️ **KNOWN** | failed today AND in the prior nightly, "
                 f"not yet chronic | {n_known} |")
    lines.append(f"| 🆕 **NEW** | first time seen "
                 f"(no prior nightly to compare) | {n_new} |")
    lines.append("")

    def _table(title: str, rows: list[tuple[str, str]]) -> None:
        if not rows:
            return
        lines.append(f"### {title} ({len(rows)})")
        lines.append("")
        by_node: dict[str, list[str]] = defaultdict(list)
        for nodeid, cell in rows:
            by_node[nodeid].append(cell)
        lines.append("| nodeid | cells affected |")
        lines.append("|---|---|")
        for nodeid, cells in sorted(by_node.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| `{nodeid}` | {', '.join(sorted(cells))} |")
        lines.append("")

    _table("🚨 REGRESSION (chronic in nightly + failing in continuous CI)",
           rr["regression"])
    _table("♻️ CHRONIC (chronic in nightly, but passing in continuous CI)",
           rr["chronic"])
    _table("🕒 CHRONIC_PENDING (chronic in nightly, no continuous evidence yet)",
           rr["chronic_pending"])
    _table("🆕 NEWLY_BROKEN (passed yesterday, failed today)", rr["newly_broken"])
    _table("♻️ KNOWN (also failing in prior nightly, not yet chronic)",
           rr["known"])
    _table("🆕 NEW (first nightly we've recorded for this test)", rr["new"])


def _bucket_counter(jobs: list[JobAnalysis]) -> Counter:
    c = Counter()
    for j in jobs:
        for f in j.failures:
            c[f.bucket] += 1
    return c


def _matrix_index(jobs: list[JobAnalysis]) -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
    """Build a dense (gpu_config x py) matrix of failure counts.

    The keys come from `matrix_cell = "<gpu>-py<py>-rocm<rocm>"`.
    """
    rows = sorted({j.matrix_cell.split("-")[0] for j in jobs})  # gpu configs
    cols = sorted({j.matrix_cell.split("-")[1] for j in jobs})  # py versions
    grid: dict[tuple[str, str], int] = {}
    for j in jobs:
        parts = j.matrix_cell.split("-")
        if len(parts) < 2:
            continue
        gpu, py = parts[0], parts[1]
        grid[(gpu, py)] = grid.get((gpu, py), 0) + len(j.failures)
    return rows, cols, grid


def render_markdown(*, run_meta: dict, jobs: list[JobAnalysis],
                    diff: dict,
                    regression_result: dict | None = None) -> str:
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
    lines.append(f"- Diff window: {diff['window_days']} prior nights "
                 f"(chronic threshold: {diff['chronic_threshold']}-of-{diff['window_days']})")
    lines.append("")

    # ---- REGRESSION classification (top of report -- highest priority) ----
    if regression_result is not None:
        _render_regression_section(lines, regression_result)

    # ---- Bucket distribution ----
    counter = _bucket_counter(jobs)
    if counter:
        lines.append("## Failure buckets")
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|---|---:|")
        for b in BUCKETS:
            n = counter.get(b, 0)
            if n:
                lines.append(f"| {b} | {n} |")
        lines.append("")

    # ---- Diff sections ----
    def _table(name: str, rows: list[tuple[str, str]], emoji: str) -> None:
        if not rows:
            return
        lines.append(f"## {emoji} {name} ({len(rows)})")
        lines.append("")
        # Group by nodeid to show blast radius as a comma-list of cells.
        by_node: dict[str, list[str]] = defaultdict(list)
        for nodeid, cell in rows:
            by_node[nodeid].append(cell)
        lines.append("| nodeid | cells affected |")
        lines.append("|---|---|")
        for nodeid, cells in sorted(by_node.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| `{nodeid}` | {', '.join(sorted(cells))} |")
        lines.append("")

    _table("Regressions (NEW today)", diff["new"], "🆕")
    _table("Recurring / chronic", diff["recurring"], "♻️")
    _table("Flaky", diff["flaky"], "⚠️")
    _table("Recovered", diff["recovered"], "✅")

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

    # ---- Per-job summary (failures, infra events, link to log) ----
    lines.append("## Per-job summary")
    lines.append("")
    for j in sorted(jobs, key=lambda x: x.matrix_cell):
        url = f"https://github.com/{run_meta.get('repo','jax-ml/jax')}/actions/runs/{run_meta['run_id']}/job/{j.job_id}"
        status = "❌" if j.conclusion == "failure" else ("✅" if j.conclusion == "success" else "⚠️")
        lines.append(f"### {status} `{j.matrix_cell}` — [{j.job_id}]({url})  ({j.duration_s}s)")
        if j.infra_events:
            lines.append(f"- infra events: {', '.join(j.infra_events)}")
        if not j.failures:
            lines.append("- no test failures parsed")
        else:
            lines.append(f"- {len(j.failures)} failures:")
            # Show up to 10 to keep the report scannable.
            for f in j.failures[:10]:
                summary = f.summary[:120].replace("\n", " ") if f.summary else ""
                lines.append(f"  - `[{f.bucket}]` `{f.nodeid}` — {summary}")
            if len(j.failures) > 10:
                lines.append(f"  - ... and {len(j.failures) - 10} more")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML (matrix heatmap, color-coded)
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
</style>
</head><body>
<h1>JAX nightly Pytest-ROCm triage — {date}</h1>
<p>
  Run <a href="{url}">{run_id}</a> · {head_sha} ·
  conclusion: <b>{conclusion}</b> ·
  jobs: {n_jobs} ({n_failed} failed, {n_passed} passed)
</p>
<p class="small">Diff window: {window} prior nights · chronic threshold: {threshold} of {window}.</p>

{regression_block}

<h2>Failure-count matrix</h2>
{matrix_table}

<h2>Diff vs prior nights</h2>
{diff_tables}

<h2>Bucket distribution</h2>
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


def render_html(*, run_meta: dict, jobs: list[JobAnalysis],
                diff: dict,
                regression_result: dict | None = None) -> str:
    rows_, cols_, grid = _matrix_index(jobs)
    matrix_html = ['<table><tr><th></th>'] + [f"<th>{c}</th>" for c in cols_] + ['</tr>']
    for r in rows_:
        matrix_html.append(f"<tr><th>{r}</th>")
        for c in cols_:
            n = grid.get((r, c), 0)
            matrix_html.append(f'<td class="{_heat_class(n)}">{n}</td>')
        matrix_html.append("</tr>")
    matrix_html.append("</table>")

    counter = _bucket_counter(jobs)
    bucket_html = ['<table><tr><th>Bucket</th><th>Count</th></tr>']
    for b in BUCKETS:
        if counter.get(b, 0):
            bucket_html.append(f"<tr><td>{b}</td><td>{counter[b]}</td></tr>")
    bucket_html.append("</table>")

    def _diff_section(title: str, items: list[tuple[str, str]]) -> str:
        if not items:
            return ""
        rows_ = []
        by_node: dict[str, list[str]] = defaultdict(list)
        for n, c in items:
            by_node[n].append(c)
        rows_.append(f"<h3>{title} ({len(items)})</h3>")
        rows_.append('<table><tr><th>nodeid</th><th>cells</th></tr>')
        for nodeid, cells in sorted(by_node.items(), key=lambda kv: -len(kv[1])):
            rows_.append(f'<tr><td class="nodeid">{nodeid}</td>'
                         f'<td>{", ".join(sorted(cells))}</td></tr>')
        rows_.append("</table>")
        return "\n".join(rows_)

    diff_html = "\n".join(filter(None, [
        _diff_section("🆕 New (regressions)", diff["new"]),
        _diff_section("♻️ Recurring / chronic", diff["recurring"]),
        _diff_section("⚠️ Flaky", diff["flaky"]),
        _diff_section("✅ Recovered", diff["recovered"]),
    ])) or "<p>No prior nights to diff against.</p>"

    # Regression-classification block (rendered above the chronic diff).
    regression_html_parts: list[str] = []
    if regression_result is not None:
        rr = regression_result
        prior     = rr.get("prior_nightly_run_id")
        cont_n    = len(rr.get("continuous_runs_used") or [])
        n_reg     = len(rr["regression"])
        n_chronic = len(rr["chronic"])
        n_cpend   = len(rr["chronic_pending"])
        n_newly   = len(rr["newly_broken"])
        n_known   = len(rr["known"])
        n_new     = len(rr["new"])
        window    = rr.get("window_days", 7)
        threshold = rr.get("chronic_threshold", 4)
        regression_html_parts.append(
            f"<h2>🚨 Regression check "
            f"(chronic ≥{threshold}-of-{window} + continuous CI)</h2>"
        )
        regression_html_parts.append("<p>")
        regression_html_parts.append(
            f"Immediately prior nightly: "
            f"{'<code>' + str(prior) + '</code>' if prior else '<em>none in DB</em>'}. "
            f"Chronic window: <b>{window}</b> nights, threshold "
            f"<b>≥{threshold}</b> of <b>{window}</b>. "
            f"Continuous-CI evidence: <b>{cont_n}</b> run(s).")
        regression_html_parts.append("</p>")
        regression_html_parts.append(
            "<table>"
            "<tr><th>Bucket</th><th>Count</th></tr>"
            f"<tr><td>🚨 REGRESSION</td><td>{n_reg}</td></tr>"
            f"<tr><td>♻️ CHRONIC</td><td>{n_chronic}</td></tr>"
            f"<tr><td>🕒 CHRONIC_PENDING</td><td>{n_cpend}</td></tr>"
            f"<tr><td>🆕 NEWLY_BROKEN</td><td>{n_newly}</td></tr>"
            f"<tr><td>♻️ KNOWN</td><td>{n_known}</td></tr>"
            f"<tr><td>🆕 NEW</td><td>{n_new}</td></tr>"
            "</table>")
        regression_html_parts.append(_diff_section(
            "🚨 REGRESSION (chronic in nightly + failing in continuous CI)",
            rr["regression"]))
        regression_html_parts.append(_diff_section(
            "♻️ CHRONIC (chronic in nightly, but passing in continuous CI)",
            rr["chronic"]))
        regression_html_parts.append(_diff_section(
            "🕒 CHRONIC_PENDING (chronic in nightly, no continuous evidence yet)",
            rr["chronic_pending"]))
        regression_html_parts.append(_diff_section(
            "🆕 NEWLY_BROKEN (passed yesterday, failed today)",
            rr["newly_broken"]))
        regression_html_parts.append(_diff_section(
            "♻️ KNOWN (also failing in prior nightly, not yet chronic)",
            rr["known"]))
        regression_html_parts.append(_diff_section(
            "🆕 NEW (first nightly we've recorded for this test)", rr["new"]))

    per_job_html = []
    for j in sorted(jobs, key=lambda x: x.matrix_cell):
        url = f"https://github.com/{run_meta.get('repo','jax-ml/jax')}/actions/runs/{run_meta['run_id']}/job/{j.job_id}"
        per_job_html.append(f'<h3>{j.matrix_cell} '
                            f'<a class="small" href="{url}">job {j.job_id}</a></h3>')
        if j.infra_events:
            per_job_html.append("<p>" + "".join(
                f'<span class="pill">{ev}</span>' for ev in j.infra_events) + "</p>")
        if j.failures:
            per_job_html.append('<table><tr><th>bucket</th><th>nodeid</th><th>summary</th></tr>')
            for f in j.failures[:50]:
                per_job_html.append(
                    f'<tr><td>{f.bucket}</td>'
                    f'<td class="nodeid">{f.nodeid}</td>'
                    f'<td>{(f.summary or "")[:200]}</td></tr>')
            per_job_html.append("</table>")
            if len(j.failures) > 50:
                per_job_html.append(f'<p class="small">... and {len(j.failures)-50} more</p>')
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
        window=diff["window_days"],
        threshold=diff["chronic_threshold"],
        regression_block="\n".join(filter(None, regression_html_parts)),
        matrix_table="\n".join(matrix_html),
        bucket_table="\n".join(bucket_html),
        diff_tables=diff_html,
        per_job="\n".join(per_job_html),
    )


# ---------------------------------------------------------------------------
# Convenience: write all three side by side
# ---------------------------------------------------------------------------

def write_all(out_dir: Path, *, run_meta: dict,
              jobs: list[JobAnalysis], diff: dict,
              regression_result: dict | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json":     out_dir / "summary.json",
        "markdown": out_dir / "report.md",
        "html":     out_dir / "report.html",
    }
    # Always write UTF-8: report.md/report.html contain emoji (🚨/⚠️/♻️/🆕)
    # that Python's locale-default codec on Windows (cp1252) cannot encode.
    paths["json"].write_text(render_json(
        run_meta=run_meta, jobs=jobs, diff=diff,
        regression_result=regression_result), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(
        run_meta=run_meta, jobs=jobs, diff=diff,
        regression_result=regression_result), encoding="utf-8")
    paths["html"].write_text(render_html(
        run_meta=run_meta, jobs=jobs, diff=diff,
        regression_result=regression_result), encoding="utf-8")
    return paths
