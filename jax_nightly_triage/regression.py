"""SQLite-backed history of failures + the six-bucket classifier.

Schema is small: five tables (``runs``, ``jobs``, ``failures``, ``flakies``,
``infra_events``), indexed on the columns we actually query (``date``,
``nodeid``, ``run_id``, ``bucket``).  The on-disk file lives under
``reports/history.db`` by default.

The single classification entry point is :func:`regression_classify`, which
applies a two-stage triage rule against (a) the prior nightlies in a
rolling window and (b) any continuous-CI runs that completed strictly
after the latest (today's) nightly.

Six buckets, in priority order (top wins):

    cancelled_infra  -- per-cell.  The latest-nightly job for
                        ``(gpu, py, rocm)`` produced no pytest pass/fail
                        signal: conclusion was cancelled / timed_out, OR
                        the job concluded "failure" with infra_events
                        recorded and zero parsed failures.  Per-test
                        buckets are skipped for these cells.

    flaky            -- per-test.  Either:
                          (a) ``pytest-rerunfailures`` retried the test in
                              today's job and the retry passed (test ends
                              up in ``flaky_tests``), OR
                          (b) "mixed history": failed today AND prior
                              nights show both passes and failures for
                              this ``(nodeid, gpu, py)``.

    chronic          -- per-test.  Failed today AND continuous-CI evidence
                        exists for the same ``(nodeid, gpu, py)`` AND
                        continuous-CI passed the test.  Continuous-CI is
                        py3.11-only, so only py3.11 nightly cells can ever
                        land here.  (This bucket is what the spec calls
                        "latest-continuous"; we keep the historical name
                        ``chronic`` for compatibility with downstream
                        consumers.)

    regression       -- per-test.  Failed today AND has FULL prior history
                        in the window (job ran every prior night) AND
                        passed in all of those prior nights.

    known            -- per-test.  Failed today AND failed in every prior
                        night where the job for ``(gpu, py)`` ran (>= 1
                        data point), AND the test isn't covered by a
                        passing continuous run.

    newly-failed       -- per-test.  Failed today AND either:
                          * no prior nightly job ran for ``(gpu, py)`` in
                            the window (the cell itself is new), OR
                          * partial prior history (J < N nights ran) AND
                            this nodeid never failed in any of them
                            (the test is new in the cell, or the cell
                            joined mid-window and the test happened to
                            pass).

For ``regression`` / ``known``, if a continuous run exists for
``(gpu, py)`` and it ALSO failed the test, the per-test record is tagged
``+continuous`` (a stronger signal for reviewers).  This does NOT move the
test between buckets -- it is rendered as a badge in the report.

Match keys
----------

The per-test buckets ignore the ROCm tag and key on ``(nodeid, gpu, py)``.
A ROCm-tag bump between nights does not wipe history.  The Stage-2
continuous-CI cross-check uses the SAME key.  Continuous-CI runs only
publish ``py3.11``, so ``py3.12 / py3.13 / py3.14`` nightly cells are
Stage-1-only by construction.

Storage limitation
------------------

The DB only stores *failures*, not the full set of executed tests.  We
infer "the test ran in night N" by observing that the JOB for
``(gpu, py)`` ran in night N (i.e. a row exists in ``jobs``) and the
test is absent from ``failures`` for that job.  This conflates
"test ran and passed" with "test didn't exist yet but the job ran".
For most JAX tests the collected set is stable across a week, so the
conflation is rare; the precise fix is to start tracking executed
nodeids per job (separate table, larger DB).  Surfaced here so future
maintainers know why ``regression`` and ``newly-failed`` are
distinguished only by job-coverage (full vs. partial) rather than by
true test-existence.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
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

CREATE TABLE IF NOT EXISTS flakies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(job_id),
    nodeid      TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_flakies_nodeid    ON flakies(nodeid);
CREATE INDEX IF NOT EXISTS idx_flakies_job       ON flakies(job_id);
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
        c.execute("DELETE FROM flakies  WHERE job_id IN (SELECT job_id FROM jobs WHERE run_id = ?)", (run_id,))
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
            for nodeid in j.flaky_tests:
                c.execute("""
                    INSERT OR IGNORE INTO flakies(job_id, nodeid)
                    VALUES (?, ?)
                """, (j.job_id, nodeid))
            for ev in j.infra_events:
                c.execute("INSERT INTO infra_events(job_id, event) VALUES (?, ?)",
                          (j.job_id, ev))


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
# Matrix-cell axis helpers
# ---------------------------------------------------------------------------
#
# matrix_cell is canonicalized as "<gpu>-py<py>-rocm<rocm>", e.g.
# "1gpu-py3.11-rocm7.2.0".  We key history and continuous-CI matching on
# (gpu, py) only -- ROCm-tag bumps don't wipe history.

_CELL_PARTS_RE = re.compile(
    r"^(?P<gpu>\d+gpu)-(?P<py>py[\d.]+)(?:-rocm[\d.]+)?$",
    re.I,
)


def gpu_py_of(matrix_cell: str) -> tuple[str, str]:
    """Return ``(gpu, py)`` extracted from a canonical matrix_cell.

    Examples:

        >>> gpu_py_of("1gpu-py3.11-rocm7.2.0")
        ('1gpu', 'py3.11')
        >>> gpu_py_of("4gpu-py3.13-rocm7.3.1")
        ('4gpu', 'py3.13')

    Falls back to ``(matrix_cell, "")`` for cells we can't parse so they
    still get a stable bucket key (matching themselves only).
    """
    m = _CELL_PARTS_RE.match(matrix_cell or "")
    if not m:
        return (matrix_cell or "", "")
    return (m.group("gpu").lower(), m.group("py").lower())


# ---------------------------------------------------------------------------
# Six-bucket classification
# ---------------------------------------------------------------------------

# Buckets used in the JSON / markdown / HTML output.
BUCKETS = (
    "cancelled_infra",
    "flaky",
    "chronic",
    "regression",
    "known",
    "newly-failed",
)


def _runs_in_window(c: sqlite3.Connection, *,
                    today_created_at: str,
                    window_days: int,
                    workflow_name_re: str,
                    end_inclusive: bool = False) -> list[sqlite3.Row]:
    """Return runs of the given workflow created in
    ``[today_created_at - window_days, today_created_at)``.

    Used for both the prior-nightly history window and (with a different
    workflow regex) the post-today continuous window via
    :func:`_continuous_runs_after`.  ``end_inclusive=True`` includes the
    today run itself; default False excludes it.
    """
    today_dt = datetime.fromisoformat(
        today_created_at.replace("Z", "+00:00"))
    start_iso = (today_dt - timedelta(days=window_days)).isoformat()
    op = "<=" if end_inclusive else "<"
    rows = c.execute(f"""
        SELECT run_id, workflow_name, created_at, date
        FROM runs
        WHERE created_at >= ? AND created_at {op} ?
        ORDER BY created_at ASC
    """, (start_iso, today_created_at)).fetchall()
    pat = re.compile(workflow_name_re)
    return [r for r in rows if pat.search(r["workflow_name"] or "")]


def _continuous_runs_after(c: sqlite3.Connection, *,
                           today_created_at: str,
                           workflow_name_re: str) -> list[int]:
    """Continuous-CI runs created strictly *after* the latest nightly.

    The spec is explicit: Stage-2 evidence comes from continuous runs
    that landed AFTER today's nightly.  Older continuous runs are
    ignored.  Returns sorted run ids.
    """
    rows = c.execute("""
        SELECT run_id, workflow_name
        FROM runs
        WHERE created_at > ?
        ORDER BY created_at ASC
    """, (today_created_at,)).fetchall()
    pat = re.compile(workflow_name_re)
    return [int(r["run_id"]) for r in rows
            if pat.search(r["workflow_name"] or "")]


def _job_cells_per_run(c: sqlite3.Connection,
                       run_ids: list[int]) -> dict[int, set[tuple[str, str]]]:
    """Return ``{run_id: {(gpu, py), ...}}`` -- which (gpu, py) cells the
    given runs executed *with a usable pytest signal*.

    A cell is "usable" iff ``conclusion in {success, failure}``.  Cells
    that were cancelled / timed_out / errored before pytest produced a
    summary are excluded from coverage counts -- we never record "passed"
    or "failed" for them.
    """
    if not run_ids:
        return {}
    placeholders = ",".join("?" * len(run_ids))
    rows = c.execute(f"""
        SELECT run_id, matrix_cell, conclusion
        FROM jobs
        WHERE run_id IN ({placeholders})
    """, run_ids).fetchall()
    out: dict[int, set[tuple[str, str]]] = {}
    for r in rows:
        if (r["conclusion"] or "") not in ("success", "failure"):
            continue
        gp = gpu_py_of(r["matrix_cell"])
        out.setdefault(int(r["run_id"]), set()).add(gp)
    return out


def _failures_keyed_gpu_py(c: sqlite3.Connection,
                           run_ids: list[int]
                           ) -> set[tuple[str, str, str]]:
    """Return ``{(nodeid, gpu, py)}`` failed across ``run_ids``."""
    if not run_ids:
        return set()
    placeholders = ",".join("?" * len(run_ids))
    rows = c.execute(f"""
        SELECT f.nodeid, j.matrix_cell
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        WHERE j.run_id IN ({placeholders})
    """, run_ids).fetchall()
    out: set[tuple[str, str, str]] = set()
    for r in rows:
        gpu, py = gpu_py_of(r["matrix_cell"])
        out.add((r["nodeid"], gpu, py))
    return out


def _failures_per_run_keyed_gpu_py(
        c: sqlite3.Connection,
        run_ids: list[int],
        ) -> dict[int, set[tuple[str, str, str]]]:
    """``{run_id: {(nodeid, gpu, py), ...}}`` for the given runs."""
    if not run_ids:
        return {}
    placeholders = ",".join("?" * len(run_ids))
    rows = c.execute(f"""
        SELECT j.run_id, f.nodeid, j.matrix_cell
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        WHERE j.run_id IN ({placeholders})
    """, run_ids).fetchall()
    out: dict[int, set[tuple[str, str, str]]] = {}
    for r in rows:
        gpu, py = gpu_py_of(r["matrix_cell"])
        out.setdefault(int(r["run_id"]), set()).add((r["nodeid"], gpu, py))
    return out


def _today_jobs_with_cell(c: sqlite3.Connection, run_id: int
                          ) -> list[sqlite3.Row]:
    """Return today's job rows (one per matrix_cell)."""
    return c.execute("""
        SELECT job_id, matrix_cell, conclusion
        FROM jobs
        WHERE run_id = ?
    """, (run_id,)).fetchall()


def _today_failures(c: sqlite3.Connection, run_id: int
                    ) -> list[tuple[str, str, str, str]]:
    """``[(nodeid, matrix_cell, gpu, py), ...]`` failed in today's run."""
    rows = c.execute("""
        SELECT f.nodeid, j.matrix_cell
        FROM failures f
        JOIN jobs j ON j.job_id = f.job_id
        WHERE j.run_id = ?
    """, (run_id,)).fetchall()
    out: list[tuple[str, str, str, str]] = []
    for r in rows:
        gpu, py = gpu_py_of(r["matrix_cell"])
        out.append((r["nodeid"], r["matrix_cell"], gpu, py))
    return out


def _today_flakies(c: sqlite3.Connection, run_id: int
                   ) -> set[tuple[str, str, str, str]]:
    """``{(nodeid, matrix_cell, gpu, py), ...}`` rerun-passed in today's
    run -- i.e. tests that pytest-rerunfailures retried and that ended
    up PASSED.
    """
    rows = c.execute("""
        SELECT fl.nodeid, j.matrix_cell
        FROM flakies fl
        JOIN jobs j ON j.job_id = fl.job_id
        WHERE j.run_id = ?
    """, (run_id,)).fetchall()
    out: set[tuple[str, str, str, str]] = set()
    for r in rows:
        gpu, py = gpu_py_of(r["matrix_cell"])
        out.add((r["nodeid"], r["matrix_cell"], gpu, py))
    return out


def _today_infra_cells(c: sqlite3.Connection, run_id: int
                       ) -> dict[str, dict]:
    """Return cells that produced *no usable pytest signal* in today's run.

    A cell is cancelled_infra if EITHER:

      * conclusion is in {cancelled, timed_out, skipped}, OR
      * conclusion is failure AND there are infra_events recorded AND
        zero failures parsed.

    Returns ``{matrix_cell: {"reason": str, "events": [...]}}``.
    """
    rows = c.execute("""
        SELECT j.job_id, j.matrix_cell, j.conclusion,
               (SELECT COUNT(*) FROM failures f WHERE f.job_id = j.job_id) AS n_fail,
               (SELECT GROUP_CONCAT(event, '|') FROM infra_events ie
                WHERE ie.job_id = j.job_id) AS events
        FROM jobs j
        WHERE j.run_id = ?
    """, (run_id,)).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        events = (r["events"] or "").split("|") if r["events"] else []
        events = [e for e in events if e]
        conclusion = r["conclusion"] or ""
        if conclusion in ("cancelled", "timed_out", "skipped"):
            out[r["matrix_cell"]] = {
                "reason": conclusion, "events": events,
            }
        elif conclusion == "failure" and events and not r["n_fail"]:
            out[r["matrix_cell"]] = {
                "reason": "infra_no_signal", "events": events,
            }
    return out


def regression_classify(db_path: Path, *,
                        today_run_id: int,
                        today_workflow_re: str,
                        continuous_workflow_re: str,
                        window_days: int = 7) -> dict:
    """Classify today's run into the six-bucket model documented at the
    top of this module.

    Args:
        db_path: SQLite file.
        today_run_id: the run id we're triaging.
        today_workflow_re: regex matching the nightly workflow name.
        continuous_workflow_re: regex matching the continuous workflow name.
        window_days: prior-nightly history window, in nights (default 7,
            and the spec caps it at 7 -- callers should not pass more).

    Returns dict with keys:

        ``regression``, ``known``, ``chronic``, ``flaky``, ``newly_failed``,
        ``cancelled_infra``  -- each a sorted list.  Per-test buckets are
        ``[(nodeid, matrix_cell), ...]``; ``cancelled_infra`` is
        ``[(matrix_cell, reason, [events...]), ...]``.

        ``stage2_continuous_confirmed``: subset of
        ``regression + known`` whose failure was ALSO observed in a
        post-today continuous-CI run (rendered as a ``+continuous``
        badge in the report).

        Plus traceability fields:
            ``prior_nightly_run_ids``, ``continuous_runs_used``,
            ``today_failure_count``, ``today_flaky_count``,
            ``today_cell_count``, ``window_days``.
    """
    if window_days > 7:
        window_days = 7  # spec hard cap

    ensure_schema(db_path)
    with connect(db_path) as c:
        today = c.execute(
            "SELECT run_id, created_at, date FROM runs WHERE run_id = ?",
            (today_run_id,)).fetchone()
        if not today:
            raise ValueError(f"run_id {today_run_id} not in DB")
        today_created_at = today["created_at"]

        # ---- Cells without pytest signal in latest nightly ---------------
        infra_cells = _today_infra_cells(c, today_run_id)
        today_cell_count = len(_today_jobs_with_cell(c, today_run_id))

        # ---- Today's failures and rerun-passed (flaky-by-rerun) ---------
        today_failures = _today_failures(c, today_run_id)
        today_flaky = _today_flakies(c, today_run_id)

        # ---- Prior nightly window (Stage 1) -----------------------------
        prior_runs = _runs_in_window(
            c, today_created_at=today_created_at,
            window_days=window_days,
            workflow_name_re=today_workflow_re,
            end_inclusive=False)
        prior_run_ids = [int(r["run_id"]) for r in prior_runs]
        prior_cells = _job_cells_per_run(c, prior_run_ids)
        prior_failures_by_run = _failures_per_run_keyed_gpu_py(
            c, prior_run_ids)

        # ---- Continuous-CI evidence (Stage 2) ---------------------------
        cont_run_ids = _continuous_runs_after(
            c, today_created_at=today_created_at,
            workflow_name_re=continuous_workflow_re)
        cont_cells_per_run = _job_cells_per_run(c, cont_run_ids)
        cont_failures = _failures_keyed_gpu_py(c, cont_run_ids)
        # A (gpu, py) is "covered by continuous" iff at least one
        # continuous run had a usable pytest signal for that cell.
        covered_gpu_py: set[tuple[str, str]] = set()
        for cells in cont_cells_per_run.values():
            covered_gpu_py |= cells

    # ---- Per-test classification ----------------------------------------
    regression: list[tuple[str, str]] = []
    known: list[tuple[str, str]] = []
    chronic: list[tuple[str, str]] = []
    flaky: list[tuple[str, str]] = []
    newly_failed: list[tuple[str, str]] = []
    stage2_confirmed: list[tuple[str, str]] = []

    # Skip per-test classification for cells without a pytest signal
    # today (they're already cancelled_infra).
    skip_cells = set(infra_cells.keys())

    # Compute, for each (gpu, py) cell present today, the number of prior
    # nightly runs in the window that had a usable signal for that cell.
    prior_cell_coverage: dict[tuple[str, str], int] = {}
    for cells in prior_cells.values():
        for gp in cells:
            prior_cell_coverage[gp] = prior_cell_coverage.get(gp, 0) + 1
    n_prior_runs = len(prior_run_ids)

    # Build a quick lookup: did (nodeid, gpu, py) fail in run R?
    def _failed_in(run_id: int, key: tuple[str, str, str]) -> bool:
        return key in prior_failures_by_run.get(run_id, set())

    # Today's flaky-by-rerun set (keyed by full triple) -- looked up first.
    today_flaky_keys = {(n, g, p) for (n, _c, g, p) in today_flaky}

    for nodeid, cell, gpu, py in today_failures:
        if cell in skip_cells:
            # The job didn't produce a pytest signal at all -- the
            # short-summary parse must have come from a partial log.
            # Defer to cancelled_infra and don't double-classify.
            continue

        key3 = (nodeid, gpu, py)

        # 1. flaky-by-rerun -- highest priority, log-local signal.
        if key3 in today_flaky_keys:
            flaky.append((nodeid, cell))
            continue

        # 2. chronic = (gpu, py) covered by continuous AND continuous
        #    PASSED this test (i.e. test absent from continuous_failures).
        cont_covers = (gpu, py) in covered_gpu_py
        cont_failed = key3 in cont_failures
        if cont_covers and not cont_failed:
            chronic.append((nodeid, cell))
            continue

        # 3-5. Stage-1 history (apply over the runs where the cell ran).
        runs_ran_cell = [r for r in prior_run_ids
                         if (gpu, py) in prior_cells.get(r, set())]
        n_ran = len(runs_ran_cell)
        n_failed = sum(1 for r in runs_ran_cell if _failed_in(r, key3))
        n_passed = n_ran - n_failed

        if n_ran == 0:
            # Job for (gpu, py) never ran in window -> brand-new cell.
            newly_failed.append((nodeid, cell))
            continue
        if n_failed == 0:
            # Never failed in any prior night the cell ran.
            if n_ran == n_prior_runs and n_prior_runs > 0:
                # Full coverage AND all-passed -> regression (today is the
                # first failure on a previously-green test).
                regression.append((nodeid, cell))
            else:
                # Partial coverage with no failures -> newly-failed (we
                # cannot say "passed in all 7" because we don't have 7).
                newly_failed.append((nodeid, cell))
            if cont_covers and cont_failed:
                stage2_confirmed.append((nodeid, cell))
            continue
        if n_passed == 0:
            # Failed every prior night the cell ran -> known.
            known.append((nodeid, cell))
            if cont_covers and cont_failed:
                stage2_confirmed.append((nodeid, cell))
            continue

        # 6. Mixed prior history -> statistical flaky.
        flaky.append((nodeid, cell))

    cancelled_infra = sorted(
        (cell, meta["reason"], tuple(meta["events"]))
        for cell, meta in infra_cells.items()
    )

    return {
        "regression":      sorted(regression),
        "known":           sorted(known),
        "chronic":         sorted(chronic),
        "flaky":           sorted(flaky),
        "newly_failed":      sorted(newly_failed),
        "cancelled_infra": cancelled_infra,
        "stage2_continuous_confirmed": sorted(stage2_confirmed),
        "prior_nightly_run_ids":  prior_run_ids,
        "continuous_runs_used":   cont_run_ids,
        "today_failure_count":    len(today_failures),
        "today_flaky_count":      len(today_flaky),
        "today_cell_count":       today_cell_count,
        "window_days":            window_days,
    }
