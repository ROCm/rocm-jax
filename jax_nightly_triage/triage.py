#!/usr/bin/env python3
"""Generic GH Actions nightly triage pipeline.

The defaults (`jax-ml/jax`, `Pytest ROCm` jobs) are tuned for JAX, but every
piece is overridable via CLI flag or environment variable, so any team that
runs pytest in GitHub Actions can drop the directory in and use it as-is.

Auth: set ``GITHUB_TOKEN`` (or use ``gh auth login``).  See README.md or run
``python3 github_client.py whoami`` for auth diagnostics.

Subcommands::

    triage.py run            # default: triage today's nightly + write report
    triage.py workflows      # list workflows in --repo (find name regex)
    triage.py runs           # list recent runs of the resolved workflow
    triage.py whoami         # show which auth path is in use
    triage.py rate-limit     # show GitHub rate-limit headroom

Env-var overrides (all optional)::

    GITHUB_TOKEN / GH_TOKEN  PAT or fine-grained token (preferred)
    TRIAGE_REPO              default --repo
    TRIAGE_WORKFLOW_RE       default --workflow-name-re
    TRIAGE_JOB_PREFIX        default --job-prefix
    TRIAGE_BRANCH            default --branch
    TRIAGE_WINDOW_DAYS       default --window-days
    TRIAGE_CHRONIC_THRESHOLD default --chronic-threshold
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import analyze_job
import regression
import report
from github_client import GitHubAuthError, GitHubClient


# ---------------------------------------------------------------------------
# Defaults (overridable via env vars and CLI flags)
# ---------------------------------------------------------------------------

DEFAULT_REPO              = os.environ.get("TRIAGE_REPO", "jax-ml/jax")
DEFAULT_WORKFLOW_NAME_RE  = os.environ.get(
    "TRIAGE_WORKFLOW_RE", r"Wheel Tests \(Nightly/Release\)")
DEFAULT_CONTINUOUS_RE     = os.environ.get(
    "TRIAGE_CONTINUOUS_RE", r"Wheel Tests \(Continuous\)")
DEFAULT_JOB_NAME_PREFIX   = os.environ.get("TRIAGE_JOB_PREFIX", "Pytest ROCm")
DEFAULT_BRANCH            = os.environ.get("TRIAGE_BRANCH", "main")
DEFAULT_WINDOW_DAYS       = int(os.environ.get("TRIAGE_WINDOW_DAYS", "7"))
DEFAULT_CHRONIC_THRESHOLD = int(os.environ.get("TRIAGE_CHRONIC_THRESHOLD", "4"))
DEFAULT_DB                = Path(os.environ.get("TRIAGE_DB", "reports/history.db"))
DEFAULT_REPORTS_DIR       = Path(os.environ.get("TRIAGE_REPORTS_DIR", "reports"))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_workflow_id(client: GitHubClient, repo: str, name_re: str) -> int:
    workflows = client.list_workflows(repo)
    pat = re.compile(name_re)
    matches = [w for w in workflows if pat.search(w["name"])]
    if not matches:
        raise SystemExit(
            f"No workflow matching /{name_re}/ in {repo}.  "
            f"Run `python3 triage.py workflows` to list available names.")
    if len(matches) > 1:
        names = "\n".join(f"  {w['id']:>10}  {w['name']}" for w in matches)
        print(f"warning: multiple workflow matches; using first.\n{names}",
              file=sys.stderr)
    return int(matches[0]["id"])


def latest_completed_run(client: GitHubClient, repo: str, workflow_id: int,
                         *, branch: str) -> dict:
    for r in client.list_runs(repo, workflow_id, branch=branch, per_page=20):
        if r.get("status") == "completed":
            return r
    raise SystemExit("No completed runs found in the recent window.")


def list_target_jobs(client: GitHubClient, repo: str, run_id: int,
                     *, name_prefix: str) -> list[dict]:
    return [j for j in client.list_jobs(repo, run_id)
            if j.get("name", "").startswith(name_prefix)]


# ---------------------------------------------------------------------------
# Per-job analysis (parallel)
# ---------------------------------------------------------------------------

def analyze_jobs_parallel(jobs: list[dict], *, repo: str, log_dir: Path,
                          client: GitHubClient,
                          max_workers: int = 12,
                          ) -> list[analyze_job.JobAnalysis]:
    out: list[analyze_job.JobAnalysis] = []
    if not jobs:
        return out

    def _do(job: dict) -> analyze_job.JobAnalysis:
        try:
            return analyze_job.analyze(job, repo=repo, log_dir=log_dir,
                                       client=client)
        except Exception as e:
            stub = analyze_job.JobAnalysis(
                job_id=int(job["id"]), name=job.get("name", "?"),
                matrix_cell=analyze_job.parse_matrix_cell(job.get("name", "?")),
                conclusion=job.get("conclusion") or "unknown",
                duration_s=0,
            )
            stub.infra_events.append(f"ANALYZER_ERROR: {type(e).__name__}: {e}")
            return stub

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_do, j): j for j in jobs}
        for fut in cf.as_completed(futures):
            a = fut.result()
            out.append(a)
            n_fail = len(a.failures)
            print(f"  [{a.matrix_cell}] {a.conclusion} -- "
                  f"{n_fail} failures, {len(a.infra_events)} infra events",
                  file=sys.stderr)
    return sorted(out, key=lambda a: a.matrix_cell)


# ---------------------------------------------------------------------------
# Subcommand: run (the main pipeline)
# ---------------------------------------------------------------------------

def _run_date(run: dict) -> str:
    return datetime.fromisoformat(
        run["created_at"].replace("Z", "+00:00")
    ).date().isoformat()


def _triage_one_run(run: dict, *, args: argparse.Namespace,
                    client: GitHubClient,
                    job_prefix: Optional[str] = None,
                    label: str = "run",
                    ) -> tuple[dict, list[analyze_job.JobAnalysis]]:
    """Pull jobs + logs for one run and return (run_meta, analyses)."""
    run_id = int(run["id"])
    run_date = _run_date(run)
    prefix = job_prefix or args.job_prefix
    print(f"[{label}] {run_id} ({run.get('name')}, {run_date}, "
          f"{run.get('conclusion')})", file=sys.stderr)
    jobs_meta = list_target_jobs(client, args.repo, run_id, name_prefix=prefix)
    if not jobs_meta:
        print(f"  no jobs match prefix {prefix!r}", file=sys.stderr)
        return ({}, [])

    log_dir = args.reports_dir / run_date / "logs"
    analyses = analyze_jobs_parallel(jobs_meta, repo=args.repo,
                                     log_dir=log_dir, client=client,
                                     max_workers=args.workers)
    run_meta = {
        "run_id": run_id,
        "date": run_date,
        "head_sha": run.get("head_sha"),
        "conclusion": run.get("conclusion"),
        "html_url": run.get("html_url"),
        "created_at": run.get("created_at"),
        "workflow_name": run.get("name"),
        "repo": args.repo,
    }
    return run_meta, analyses


def _store(run_meta: dict, analyses, *, db: Path) -> None:
    if not run_meta:
        return
    regression.store_run(
        db,
        run_id=run_meta["run_id"],
        workflow_name=run_meta["workflow_name"],
        head_sha=run_meta["head_sha"] or "",
        run_date=run_meta["date"],
        created_at=run_meta["created_at"] or "",
        conclusion=run_meta["conclusion"] or "unknown",
        html_url=run_meta["html_url"] or "",
        jobs=analyses,
    )


def _ingest_continuous_after(client: GitHubClient, *,
                             args: argparse.Namespace,
                             nightly_created_at: str) -> int:
    """Triage and store all *completed* continuous runs created strictly
    after the given nightly timestamp.  Returns count ingested.

    Skips runs already in the DB so re-runs are cheap.
    """
    try:
        cont_wf_id = find_workflow_id(client, args.repo,
                                      args.continuous_workflow_re)
    except SystemExit as e:
        print(f"  cross-check disabled: {e}", file=sys.stderr)
        return 0

    runs = client.list_runs(args.repo, cont_wf_id, branch=args.branch,
                            per_page=args.continuous_max_runs)
    candidates = [r for r in runs
                  if r.get("status") == "completed"
                  and (r.get("created_at") or "") > nightly_created_at]
    if not candidates:
        print("  no continuous runs after nightly yet", file=sys.stderr)
        return 0

    # Skip ones already in the DB.
    regression.ensure_schema(args.db)
    with regression.connect(args.db) as c:
        existing = {row["run_id"] for row in
                    c.execute("SELECT run_id FROM runs").fetchall()}
    todo = [r for r in candidates if int(r["id"]) not in existing]
    print(f"  {len(candidates)} continuous run(s) after nightly "
          f"({len(candidates) - len(todo)} already cached, "
          f"{len(todo)} to ingest)", file=sys.stderr)

    n = 0
    for r in todo:
        meta, analyses = _triage_one_run(
            r, args=args, client=client,
            job_prefix=args.continuous_job_prefix,
            label="continuous")
        _store(meta, analyses, db=args.db)
        n += 1
    return n


def cmd_run(args: argparse.Namespace, client: GitHubClient) -> int:
    if args.run_id:
        run = client.get_run(args.repo, args.run_id)
    else:
        wf_id = find_workflow_id(client, args.repo, args.workflow_name_re)
        run = latest_completed_run(client, args.repo, wf_id, branch=args.branch)

    run_meta, analyses = _triage_one_run(run, args=args, client=client,
                                         label="nightly")
    if not run_meta:
        print(f"No jobs match prefix {args.job_prefix!r} in run {run['id']}.",
              file=sys.stderr)
        print("Hint: run `python3 triage.py runs --repo <REPO>` to inspect.",
              file=sys.stderr)
        return 2

    if not args.no_store:
        _store(run_meta, analyses, db=args.db)

    # ------------------------------------------------------------------
    # Cross-check against continuous CI to elevate confidence.
    # ------------------------------------------------------------------
    if args.cross_check_continuous and not args.no_store:
        ingested = _ingest_continuous_after(
            client, args=args,
            nightly_created_at=run_meta["created_at"])
        if ingested:
            print(f"  ingested {ingested} continuous run(s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Run BOTH classifiers:
    #   * Stage 1: rolling-window chronic diff (NEW / RECURRING / FLAKY /
    #     RECOVERED), purely from nightly history.
    #   * Stage 2: multi-source classification (REGRESSION / CHRONIC /
    #     CHRONIC_PENDING / NEWLY_BROKEN / KNOWN / NEW), which combines
    #     the chronic window with continuous-CI evidence.
    # Both go into the report; Stage 2 is rendered at the top.
    # ------------------------------------------------------------------
    diff = regression.compute_diff(
        args.db, today=run_meta["date"],
        days=args.window_days,
        chronic_threshold=args.chronic_threshold,
    )

    regression_result: Optional[dict] = None
    if args.cross_check_continuous:
        try:
            regression_result = regression.regression_classify(
                args.db,
                today_run_id=run_meta["run_id"],
                today_workflow_re=args.workflow_name_re,
                continuous_workflow_re=args.continuous_workflow_re,
                window_days=args.window_days,
                chronic_threshold=args.chronic_threshold,
            )
        except ValueError as e:
            print(f"  regression_classify skipped: {e}", file=sys.stderr)

    out_dir = args.reports_dir / run_meta["date"]
    paths = report.write_all(out_dir, run_meta=run_meta,
                             jobs=analyses, diff=diff,
                             regression_result=regression_result)
    print(f"Wrote: {paths['json']}", file=sys.stderr)
    print(f"Wrote: {paths['markdown']}", file=sys.stderr)
    print(f"Wrote: {paths['html']}", file=sys.stderr)

    if args.print_md:
        print()
        print(paths["markdown"].read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------
# Helper subcommands
# ---------------------------------------------------------------------------

def cmd_workflows(args: argparse.Namespace, client: GitHubClient) -> int:
    """List all workflows in the repo so a user can copy the right name."""
    workflows = client.list_workflows(args.repo)
    print(f"# {len(workflows)} workflow(s) in {args.repo}")
    print(f"# {'id':>12}  {'state':<8}  name")
    for w in workflows:
        print(f"  {w['id']:>12}  {w['state']:<8}  {w['name']}")
    print("\nTo target one:  --workflow-name-re 'exact phrase from name above'",
          file=sys.stderr)
    return 0


def cmd_runs(args: argparse.Namespace, client: GitHubClient) -> int:
    """List recent runs of the resolved workflow + their conclusions."""
    wf_id = find_workflow_id(client, args.repo, args.workflow_name_re)
    runs = client.list_runs(args.repo, wf_id, branch=args.branch,
                            per_page=args.limit)
    print(f"# Recent runs of workflow {wf_id} on {args.branch} (max {args.limit}):")
    print(f"# {'run_id':>12}  {'date':<10}  {'conclusion':<10}  url")
    for r in runs:
        date = (r.get("created_at") or "")[:10]
        print(f"  {r['id']:>12}  {date}  {r.get('conclusion','?'):<10}  {r['html_url']}")
    return 0


def cmd_whoami(_args: argparse.Namespace, client: GitHubClient) -> int:
    print(client.check_auth())
    return 0


def cmd_rate_limit(_args: argparse.Namespace, client: GitHubClient) -> int:
    print(json.dumps(client.rate_limit()["resources"]["core"], indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"GitHub repo (default: {DEFAULT_REPO}).")
    p.add_argument("--workflow-name-re", default=DEFAULT_WORKFLOW_NAME_RE,
                   help='Regex matched against workflow name '
                        f'(default: {DEFAULT_WORKFLOW_NAME_RE!r}).')
    p.add_argument("--job-prefix", default=DEFAULT_JOB_NAME_PREFIX,
                   help=f"Job-name prefix (default: {DEFAULT_JOB_NAME_PREFIX!r}).")
    p.add_argument("--branch", default=DEFAULT_BRANCH,
                   help=f"Branch (default: {DEFAULT_BRANCH}).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    # ---- run (default) ----
    run = sub.add_parser("run", help="Triage a nightly run (default).")
    _add_common(run)
    run.add_argument("--run-id", type=int, default=None,
                     help="Specific run id (default: discover latest).")
    run.add_argument("--workers", type=int, default=12,
                     help="Parallel per-job analyzers (default: 12).")
    run.add_argument("--db", type=Path, default=DEFAULT_DB,
                     help=f"SQLite history file (default: {DEFAULT_DB}).")
    run.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
                     help=f"Output root (default: {DEFAULT_REPORTS_DIR}).")
    run.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                     help=f"Prior nights to diff against (default: {DEFAULT_WINDOW_DAYS}).")
    run.add_argument("--chronic-threshold", type=int,
                     default=DEFAULT_CHRONIC_THRESHOLD,
                     help=f"Nights-failed-of-window required to mark a "
                          f"failure as chronic (drives Stage-1 RECURRING "
                          f"and Stage-2 REGRESSION/CHRONIC/"
                          f"CHRONIC_PENDING). "
                          f"Default: {DEFAULT_CHRONIC_THRESHOLD} of "
                          f"--window-days.")
    run.add_argument("--no-store", action="store_true",
                     help="Skip persisting to the SQLite history.")
    run.add_argument("--print-md", action="store_true",
                     help="Also print the markdown report to stdout.")
    # ---- Continuous CI cross-check (Stage 2 regression detection) ----
    cont = run.add_argument_group(
        "Continuous-CI cross-check (Stage 2 regression detection)",
        "After triaging the nightly, ingest any continuous CI runs that "
        "completed since (and any that fall inside the chronic window).  "
        "Continuous-CI evidence is what splits chronic-in-nightly "
        "failures into REGRESSION (continuous also failing), CHRONIC "
        "(continuous passing), or CHRONIC_PENDING (no continuous run "
        "landed yet).  Re-running the pipeline later picks up newly "
        "completed continuous runs cheaply and reclassifies "
        "CHRONIC_PENDING entries.")
    cont.add_argument("--cross-check-continuous",
                      dest="cross_check_continuous",
                      action="store_true", default=True,
                      help="Enable Stage 2 cross-check (default).")
    cont.add_argument("--no-cross-check-continuous",
                      dest="cross_check_continuous",
                      action="store_false",
                      help="Disable Stage 2 cross-check.")
    cont.add_argument("--continuous-workflow-re",
                      default=DEFAULT_CONTINUOUS_RE,
                      help=f"Regex for the continuous workflow name "
                           f"(default: {DEFAULT_CONTINUOUS_RE!r}).")
    cont.add_argument("--continuous-job-prefix",
                      default=None,
                      help="Job-name prefix in continuous (default: same "
                           "as --job-prefix).")
    cont.add_argument("--continuous-max-runs", type=int, default=20,
                      help="How many recent continuous runs to inspect "
                           "(default: 20).")

    # ---- workflows ----
    wf = sub.add_parser("workflows", help="List workflows in --repo.")
    _add_common(wf)

    # ---- runs ----
    rn = sub.add_parser("runs", help="List recent runs of the resolved workflow.")
    _add_common(rn)
    rn.add_argument("--limit", type=int, default=20,
                    help="How many recent runs to show (default: 20).")

    # ---- whoami / rate-limit ----
    sub.add_parser("whoami", help="Show which auth path is active.")
    sub.add_parser("rate-limit", help="Show GitHub API rate-limit headroom.")

    return p


def _force_utf8_streams() -> None:
    """Best-effort: switch stdout/stderr to UTF-8.

    The report contains emoji (🚨 / ⚠️ / ♻️ / 🆕) that Python's locale-default
    codec on Windows (cp1252) cannot encode. ``--print-md`` and the progress
    messages would otherwise raise ``UnicodeEncodeError``. ``reconfigure`` is
    a no-op when streams are already UTF-8 (e.g. on Linux/macOS), so this is
    safe everywhere; we silently ignore the case where the stream is not a
    text-IO that supports reconfiguration (e.g. captured by a test runner).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _force_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to "run" if no subcommand was given (back-compat with the old
    # `python3 triage.py --run-id ...` invocation).
    if args.cmd is None:
        # Re-parse with "run" injected so flags hit the run parser.
        return main(["run"] + (argv or sys.argv[1:]))

    # If the user didn't override --continuous-job-prefix, mirror --job-prefix.
    if getattr(args, "continuous_job_prefix", None) is None and \
       hasattr(args, "job_prefix"):
        args.continuous_job_prefix = args.job_prefix

    client = GitHubClient()
    try:
        client.check_auth()
    except GitHubAuthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    handler = {
        "run":        cmd_run,
        "workflows":  cmd_workflows,
        "runs":       cmd_runs,
        "whoami":     cmd_whoami,
        "rate-limit": cmd_rate_limit,
    }[args.cmd]
    return handler(args, client)


if __name__ == "__main__":
    raise SystemExit(main())
