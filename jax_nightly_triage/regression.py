"""SQLite-backed history of failures + two classification APIs.

Schema is tiny on purpose: four tables (``runs``, ``jobs``, ``failures``,
``infra_events``), indexed on the columns we actually query (``date``,
``nodeid``, ``run_id``, ``bucket``).  The on-disk file lives under
``reports/history.db`` by default.

Two independent classification APIs are exported:

* :func:`compute_diff` -- *Stage 1*, history-only.  For each
  ``(nodeid, matrix_cell)`` failing in today's run, partitions today's
  failures + the prior chronic set into::

      NEW       -- failed today, not seen in any of the prior N nights
      RECURRING -- failed today AND in >= chronic_threshold of prior N nights
      FLAKY     -- failed today AND in 1..chronic_threshold-1 of prior N nights
      RECOVERED -- chronic in prior nights but did NOT fail today

  Pure rolling-window arithmetic over the nightly history; no continuous-CI
  signal is consulted.  Useful as a coarse "is this getting worse / better?"
  view further down the report.

* :func:`regression_classify` -- *Stage 2*, multi-source.  Adds the
  continuous-CI cross-check on top of the chronic history and produces the
  six headline buckets used at the top of the report::

      REGRESSION       -- chronic in nightly + failing in continuous CI
      CHRONIC          -- chronic in nightly, but continuous CI passes
      CHRONIC_PENDING  -- chronic in nightly, but no continuous evidence yet
      NEWLY_BROKEN     -- failed today, passed in the immediately prior nightly
      KNOWN            -- failed today AND in the prior nightly, not yet chronic
      NEW              -- failed today, no prior nightly run is on file

  See the doc-comment block above :func:`regression_classify` for matching
  semantics (chronic uses full ``(nodeid, matrix_cell)``; continuous-CI
  uses ``(nodeid, gpu_axis)``).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date as date_cls, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from analyze_job import JobAnalysis


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY,
    workflow_name TEXT,
    head_sha      TEXT,
    date          TEXT NOT NULL,
    created_at    TEXT,
    conclusion    TEXT,
    html_url      TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id        INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id),
    name          TEXT NOT NULL,
    matrix_cell   TEXT NOT NULL,
    conclusion    TEXT,
    duration_s    INTEGER,
    exit_step     INTEGER
);

CREATE TABLE IF NOT EXISTS failures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(job_id),
    nodeid      TEXT NOT NULL,
    bucket      TEXT NOT NULL,
    summary     TEXT,
    UNIQUE(job_id, nodeid)
);

CREATE TABLE IF NOT EXISTS infra_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  INTEGER NOT NULL REFERENCES jobs(job_id),
    event   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_date         ON runs(date);
CREATE INDEX IF NOT EXISTS idx_jobs_run          ON jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_cell         ON jobs(matrix_cell);
CREATE INDEX IF NOT EXISTS idx_failures_nodeid   ON failures(nodeid);
CREATE INDEX IF NOT EXISTS idx_failures_job      ON failures(job_id);
CREATE INDEX IF NOT EXISTS idx_failures_bucket   ON failures(bucket);
CREATE INDEX IF NOT EXISTS idx_infra_job         ON infra_events(job_id);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@contextmanager
def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_schema(db_path: Path) -> None:
    with connect(db_path) as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def store_run(db_path: Path, *, run_id: int, workflow_name: str,
              head_sha: str, run_date: str, created_at: str,
              conclusion: str, html_url: str,
              jobs: Iterable[JobAnalysis]) -> None:
    """Idempotent: re-storing the same run_id replaces its rows."""
    ensure_schema(db_path)
    with connect(db_path) as c:
        c.execute("DELETE FROM failures WHERE job_id IN (SELECT job_id FROM jobs WHERE run_id = ?)", (run_id,))
        c.execute("DELETE FROM infra_events WHERE job_id IN (SELECT job_id FROM jobs WHERE run_id = ?)", (run_id,))
        c.execute("DELETE FROM jobs WHERE run_id = ?", (run_id,))
        c.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        c.execute("""
            INSERT INTO runs(run_id, workflow_name, head_sha, date, created_at, conclusion, html_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, workflow_name, head_sha, run_date, created_at, conclusion, html_url))
        for j in jobs:
            c.execute("""
                INSERT INTO jobs(job_id, run_id, name, matrix_cell, conclusion, duration_s, exit_step)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (j.job_id, run_id, j.name, j.matrix_cell, j.conclusion,
                  j.duration_s, j.exit_step))
            for f in j.failures:
                c.execute("""
                    INSERT OR IGNORE INTO failures(job_id, nodeid, bucket, summary)
                    VALUES (?, ?, ?, ?)
                """, (j.job_id, f.nodeid, f.bucket, f.summary[:1024]))
            for ev in j.infra_events:
                c.execute("INSERT INTO infra_events(job_id, event) VALUES (?, ?)",
                          (j.job_id, ev))


# ---------------------------------------------------------------------------
# Diff queries
# ---------------------------------------------------------------------------

def _failures_on(c: sqlite3.Connection, run_date: str) -> set[tuple[str, str]]:
    """Return {(nodeid, matrix_cell)} for the run on ``run_date``."""
    rows = c.execute("""
        SELECT f.nodeid, j.matrix_cell
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        JOIN runs r ON r.run_id = j.run_id
        WHERE r.date = ?
    """, (run_date,)).fetchall()
    return {(r["nodeid"], r["matrix_cell"]) for r in rows}


def _failures_in_window(c: sqlite3.Connection, *, today: str,
                        days: int) -> dict[tuple[str, str], int]:
    """Return {(nodeid, matrix_cell): nights_failed} over the prior ``days``
    nights (excluding ``today``)."""
    end = date_cls.fromisoformat(today)
    start = end - timedelta(days=days)
    rows = c.execute("""
        SELECT f.nodeid, j.matrix_cell, COUNT(DISTINCT r.date) AS n
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        JOIN runs r ON r.run_id = j.run_id
        WHERE r.date < ? AND r.date >= ?
        GROUP BY f.nodeid, j.matrix_cell
    """, (today, start.isoformat())).fetchall()
    return {(r["nodeid"], r["matrix_cell"]): r["n"] for r in rows}


def compute_diff(db_path: Path, *, today: str, days: int = 7,
                 chronic_threshold: int = 4) -> dict:
    """Stage-1 history-only partition (no continuous-CI signal).

    For every ``(nodeid, matrix_cell)`` failing today and every
    ``(nodeid, matrix_cell)`` that was chronic in the prior window,
    assign one of:

        NEW       -- failed today AND not seen in any of the prior `days`
                     nights.
        RECURRING -- failed today AND in >= `chronic_threshold` of the
                     prior `days` nights.
        FLAKY     -- failed today AND in 1..`chronic_threshold`-1 of the
                     prior `days` nights (some, but not enough to be
                     chronic).
        RECOVERED -- chronic in the prior window but did NOT fail today.

    For the multi-source headline buckets used at the top of the report
    (REGRESSION / CHRONIC / CHRONIC_PENDING / NEWLY_BROKEN / KNOWN / NEW),
    use :func:`regression_classify` instead.

    Args:
        today: ISO date (YYYY-MM-DD) of the run to analyze.
        days: window size for "prior" nights (default 7).
        chronic_threshold: how many of the prior ``days`` a failure must
            appear in to be called RECURRING / chronic (default 4 of 7).
    """
    ensure_schema(db_path)
    with connect(db_path) as c:
        today_set = _failures_on(c, today)
        prior = _failures_in_window(c, today=today, days=days)

    chronic = {k for k, n in prior.items() if n >= chronic_threshold}
    flaky_prior = {k for k, n in prior.items()
                   if 0 < n < chronic_threshold}

    new       = sorted(today_set - set(prior.keys()))
    recurring = sorted(today_set & chronic)
    flaky     = sorted((today_set & flaky_prior) - chronic)
    recovered = sorted(chronic - today_set)

    return {
        "today_count": len(today_set),
        "window_days": days,
        "chronic_threshold": chronic_threshold,
        "new": new,
        "recurring": recurring,
        "flaky": flaky,
        "recovered": recovered,
        "prior_nights_seen": prior,
    }


def list_runs(db_path: Path, *, limit: int = 30) -> list[dict]:
    ensure_schema(db_path)
    with connect(db_path) as c:
        rows = c.execute("""
            SELECT r.run_id, r.date, r.head_sha, r.conclusion,
                   COUNT(DISTINCT j.job_id) AS n_jobs,
                   SUM(CASE WHEN j.conclusion='failure' THEN 1 ELSE 0 END) AS n_failed_jobs
            FROM runs r
            LEFT JOIN jobs j ON j.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.date DESC, r.run_id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Regression classification (chronic + continuous-CI confirmation)
# ---------------------------------------------------------------------------
#
# Each (nodeid, matrix_cell) failing in today's nightly is assigned to
# exactly one of the six buckets below.  The first matching branch in the
# decision tree wins -- the tree is implemented at the bottom of
# :func:`regression_classify`.  Inputs:
#
#   nights_failed   = # distinct prior nights in
#                     [today - window_days, today) on which the same
#                     (nodeid, matrix_cell) failed.
#   is_chronic      = nights_failed >= chronic_threshold        (default 4-of-7)
#   in_continuous   = (nodeid, gpu_axis) failed in any continuous-CI run
#                     within the window OR strictly after today's nightly.
#   have_continuous = >= 1 continuous-CI run is in the window or after today
#                     (i.e. continuous-CI evidence exists at all).
#   prior_run_id    = id of the most recent nightly created strictly before
#                     today's, or None if there isn't one in the DB.
#   prior_full_set  = the set of (nodeid, matrix_cell) that failed in
#                     prior_run_id (empty if prior_run_id is None).
#
# Decision tree (in priority order):
#
#   if is_chronic:
#       if in_continuous                      -> REGRESSION
#       elif have_continuous                  -> CHRONIC
#       else                                  -> CHRONIC_PENDING
#   elif prior_run_id is None                 -> NEW
#   elif (nodeid, matrix_cell) in prior_full_set -> KNOWN
#   else                                      -> NEWLY_BROKEN
#
# Bucket meanings:
#
#   REGRESSION       -- chronic in nightly history AND also failing in
#                       continuous CI.  Multi-source confirmation: this is
#                       the actionable bucket -- "same test broken on the
#                       same GPU axis in two independent pipelines".
#
#   CHRONIC          -- chronic in nightly history, but continuous-CI
#                       evidence exists and shows the test passing.  Likely
#                       env-specific to the nightly runner image, or
#                       already fixed in HEAD; not actionable as a fresh
#                       regression.
#
#   CHRONIC_PENDING  -- chronic in nightly history, but NO continuous-CI
#                       runs landed in the cross-check window yet, so we
#                       cannot decide REGRESSION vs CHRONIC.  Re-running
#                       the pipeline after the next continuous run lands
#                       will reclassify these.
#
#   NEWLY_BROKEN     -- failed today, passed in the immediately prior
#                       nightly, not yet chronic.  This is what the
#                       *original* "REGRESSION" bucket meant before the
#                       multi-source check was added; preserved as its own
#                       category because "something just changed in HEAD"
#                       is still a useful signal even without continuous-CI
#                       confirmation.
#
#   KNOWN            -- failed today AND in the immediately prior nightly,
#                       but has not yet failed in enough nights to be
#                       chronic.  A breakage that started recently and is
#                       en route to CHRONIC_PENDING / CHRONIC / REGRESSION.
#
#   NEW              -- failed today; no prior nightly run for this
#                       workflow is in the database, so history is
#                       unavailable.  First-run-of-the-pipeline noise.
#                       (Distinct from compute_diff()'s "NEW", which means
#                       "not in any of the last N nights" with prior data
#                       known to exist.)
#
# Matching uses TWO different keys on purpose:
#
#   - Chronic check (history within nightly):
#       full (nodeid, matrix_cell).  Same test, same matrix cell -- the
#       strict definition of chronic.
#
#   - Continuous-CI cross-check:
#       (nodeid, gpu_axis), ignoring the python axis.  The continuous
#       workflow runs only py3.11 while the nightly fans out 3.11-3.14,
#       so a nightly failure on py3.13 is considered "covered" by a
#       py3.11 continuous run on the same GPU config.

import re

_CELL_GPU_RE = re.compile(r"^(\d+gpu)\b", re.I)


def _gpu_of(matrix_cell: str) -> str:
    """Return the GPU axis of a matrix cell, e.g. '1gpu' from '1gpu-py3.11-rocm7.2.0'."""
    m = _CELL_GPU_RE.match(matrix_cell or "")
    return m.group(1).lower() if m else (matrix_cell or "")


def _failures_for_runs(c: sqlite3.Connection,
                       run_ids: list[int]) -> set[tuple[str, str]]:
    """Return {(nodeid, gpu_config)} across the given run ids."""
    if not run_ids:
        return set()
    placeholders = ",".join("?" * len(run_ids))
    rows = c.execute(f"""
        SELECT f.nodeid, j.matrix_cell
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        WHERE j.run_id IN ({placeholders})
    """, run_ids).fetchall()
    return {(r["nodeid"], _gpu_of(r["matrix_cell"])) for r in rows}


def _today_failures_with_gpu(c: sqlite3.Connection,
                             run_id: int) -> list[tuple[str, str, str]]:
    """Return [(nodeid, matrix_cell, gpu_config)] for today's run."""
    rows = c.execute("""
        SELECT f.nodeid, j.matrix_cell
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        WHERE j.run_id = ?
    """, (run_id,)).fetchall()
    return [(r["nodeid"], r["matrix_cell"], _gpu_of(r["matrix_cell"])) for r in rows]


def _prior_run_id(c: sqlite3.Connection, *,
                  today_created_at: str,
                  workflow_name_re: str) -> Optional[int]:
    """Most-recent run of the same workflow created strictly before
    ``today_created_at``."""
    rows = c.execute("""
        SELECT run_id, workflow_name, created_at
        FROM runs
        WHERE created_at < ?
        ORDER BY created_at DESC
    """, (today_created_at,)).fetchall()
    pat = re.compile(workflow_name_re)
    for r in rows:
        if pat.search(r["workflow_name"] or ""):
            return int(r["run_id"])
    return None


def _continuous_runs_in_or_after_window(
        c: sqlite3.Connection, *,
        today_created_at: str,
        window_days: int,
        workflow_name_re: str) -> list[int]:
    """Continuous-CI runs whose created_at is within the chronic window
    [today - window_days, today_created_at) OR strictly after today's
    nightly.  We want both halves so REGRESSION's continuous evidence is
    aligned with the chronic-history evidence and isn't artificially
    starved during the gap before the next continuous run lands.
    """
    today_dt = datetime.fromisoformat(
        today_created_at.replace("Z", "+00:00"))
    start_iso = (today_dt - timedelta(days=window_days)).isoformat()
    rows = c.execute("""
        SELECT run_id, workflow_name, created_at
        FROM runs
        WHERE created_at >= ?
        ORDER BY created_at ASC
    """, (start_iso,)).fetchall()
    pat = re.compile(workflow_name_re)
    return [int(r["run_id"]) for r in rows
            if pat.search(r["workflow_name"] or "")]


def regression_classify(db_path: Path, *,
                        today_run_id: int,
                        today_workflow_re: str,
                        continuous_workflow_re: str,
                        window_days: int = 7,
                        chronic_threshold: int = 4) -> dict:
    """Classify today's failures into REGRESSION / CHRONIC /
    CHRONIC_PENDING / NEWLY_BROKEN / KNOWN / NEW.

    REGRESSION (the headline bucket) requires ALL of:

        1. The test failed in today's nightly run.
        2. The same (nodeid, matrix_cell) failed in at least
           ``chronic_threshold`` of the past ``window_days`` nightly runs.
        3. (nodeid, gpu_axis) failed in at least one continuous-CI run
           within the same window or after today's nightly.

    Args:
        db_path: SQLite file.
        today_run_id: the run id we're triaging.
        today_workflow_re: regex matching the nightly workflow name.
        continuous_workflow_re: regex matching the continuous workflow name.
        window_days: chronic-history window, in nights (default 7).
        chronic_threshold: how many of the past ``window_days`` nights a
            failure must appear in to be considered chronic
            (default 4-of-7; lower it to make REGRESSION more permissive).

    Returns:
        Dict with keys:
            ``regression``, ``chronic``, ``chronic_pending``,
            ``newly_broken``, ``known``, ``new`` -- each a sorted list of
            (nodeid, matrix_cell) tuples.
        Plus traceability fields:
            ``prior_nightly_run_id``, ``continuous_runs_used``,
            ``today_failure_count``, ``window_days``,
            ``chronic_threshold``.
    """
    ensure_schema(db_path)
    with connect(db_path) as c:
        today = c.execute(
            "SELECT run_id, created_at, date FROM runs WHERE run_id = ?",
            (today_run_id,)).fetchone()
        if not today:
            raise ValueError(f"run_id {today_run_id} not in DB")

        today_failures = _today_failures_with_gpu(c, today_run_id)
        if not today_failures:
            return _empty_classification(
                prior=None, continuous=[],
                window_days=window_days,
                chronic_threshold=chronic_threshold)

        # ---- Chronic history (nightly window) -----------------------------
        # {(nodeid, matrix_cell): nights_failed} over [today-window, today).
        prior_nights_seen = _failures_in_window(
            c, today=today["date"], days=window_days)

        # ---- Immediately prior nightly (NEWLY_BROKEN vs KNOWN) -----------
        prior_run_id = _prior_run_id(
            c, today_created_at=today["created_at"],
            workflow_name_re=today_workflow_re)
        prior_full_set: set[tuple[str, str]] = set()
        if prior_run_id is not None:
            rows = c.execute("""
                SELECT f.nodeid, j.matrix_cell
                FROM failures f
                JOIN jobs j ON j.job_id = f.job_id
                WHERE j.run_id = ?
            """, (prior_run_id,)).fetchall()
            prior_full_set = {(r["nodeid"], r["matrix_cell"]) for r in rows}

        # ---- Continuous-CI evidence (within window OR after today) -------
        continuous_run_ids = _continuous_runs_in_or_after_window(
            c, today_created_at=today["created_at"],
            window_days=window_days,
            workflow_name_re=continuous_workflow_re)
        continuous_failures = _failures_for_runs(c, continuous_run_ids)

    have_continuous_evidence = bool(continuous_run_ids)

    regression: list[tuple[str, str]] = []
    chronic: list[tuple[str, str]] = []
    chronic_pending: list[tuple[str, str]] = []
    newly_broken: list[tuple[str, str]] = []
    known: list[tuple[str, str]] = []
    new: list[tuple[str, str]] = []

    for nodeid, cell, gpu in today_failures:
        nights_failed = prior_nights_seen.get((nodeid, cell), 0)
        is_chronic = nights_failed >= chronic_threshold
        in_continuous = (nodeid, gpu) in continuous_failures

        if is_chronic:
            if in_continuous:
                regression.append((nodeid, cell))
            elif have_continuous_evidence:
                chronic.append((nodeid, cell))
            else:
                chronic_pending.append((nodeid, cell))
            continue

        # Not chronic.  Distinguish:
        #   * NEW          -- no prior nightly to compare against,
        #   * KNOWN        -- failed in the prior nightly too,
        #   * NEWLY_BROKEN -- passed in the prior nightly, fails now.
        if prior_run_id is None:
            new.append((nodeid, cell))
        elif (nodeid, cell) in prior_full_set:
            known.append((nodeid, cell))
        else:
            newly_broken.append((nodeid, cell))

    return {
        "regression":          sorted(regression),
        "chronic":             sorted(chronic),
        "chronic_pending":     sorted(chronic_pending),
        "newly_broken":        sorted(newly_broken),
        "known":               sorted(known),
        "new":                 sorted(new),
        "prior_nightly_run_id": prior_run_id,
        "continuous_runs_used": continuous_run_ids,
        "today_failure_count":  len(today_failures),
        "window_days":          window_days,
        "chronic_threshold":    chronic_threshold,
    }


def _empty_classification(*, prior, continuous,
                          window_days: int = 7,
                          chronic_threshold: int = 4) -> dict:
    return {
        "regression": [],
        "chronic": [],
        "chronic_pending": [],
        "newly_broken": [],
        "known": [],
        "new": [],
        "prior_nightly_run_id": prior,
        "continuous_runs_used": continuous,
        "today_failure_count": 0,
        "window_days": window_days,
        "chronic_threshold": chronic_threshold,
    }
