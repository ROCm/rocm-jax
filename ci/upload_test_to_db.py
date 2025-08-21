#!/usr/bin/env python3
"""
Upload pytest results to MySQL.
Tables:
 - ci_runs:    one row per run
 - ci_tests:   one row per unique test
 - ci_results: one row per test per run
"""

from __future__ import annotations

import os
import re
import ast
import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List, Tuple, Optional, Dict

# pylint: disable=import-error
import mysql.connector
from mysql.connector import Error as MySQLError

# -----------------------------
# Constants
# -----------------------------
TEXT_LIMIT = 250
BATCH_SIZE = 2000
SKIP_FILES = {
    "final_compiled_report.json",
    "final_compiled_report.html",
    "collect_module_log.jsonl",
    "test_abort_log.json",
    "test_abort_log.html",
    "test_seg_fault_log.json",
    "test_seg_fault_log.html",
}
DEFAULT_LABEL = "Skipped Upstream"


# -----------------------------
# Helpers
# -----------------------------
def shorten(s: Optional[str], limit: int = TEXT_LIMIT) -> Optional[str]:
    """Return string limited to `limit` characters, or None."""
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= limit else s[:limit]


def extract_skip_reason(reason: str) -> str:
    """Parse pytest skip longrepr tuple-string into its reason text.

    Example input: "('/path/test_x.py', 42, 'Skipped: some reason')"
    """
    try:
        tup = ast.literal_eval(reason)
        if isinstance(tup, tuple) and len(tup) >= 3:
            return str(tup[2])
    except Exception:
        pass
    return reason


def nodeid_parts(nodeid: str) -> Tuple[str, str, str]:
    """Split pytest nodeid into (filename, classname, test_name).

    Assumes the invariant "file::Class::test" is satisfied by all nodeids.
    """
    f, c, t = nodeid.split("::", 2)
    return f, c, t


def list_test_jsons(logs_dir: Path) -> List[Path]:
    """Return per-test JSON files under logs_dir; skip aux files."""
    return sorted(
        p
        for p in logs_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".json"
        and p.name not in SKIP_FILES
        and not p.name.endswith("last_running.json")
    )


def load_from_per_test_jsons(files: Iterable[Path]) -> Tuple[datetime, List[dict]]:
    """Load tests and set run created_at to the latest per-file 'created'.

    Each per-test JSON file must have a 'created' timestamp. We use the maximum
    of these values as the run's created_at (i.e., when the run finished).
    """
    tests: List[dict] = []
    created_values: List[float] = []

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "created" not in data:
            raise ValueError(f"Missing 'created' in {path.name}")
        created_values.append(float(data["created"]))
        tests.extend(data.get("tests", []))

    created_at = datetime.fromtimestamp(max(created_values), tz=timezone.utc)
    return created_at, tests


def extract_result_fields(
    t: dict,
) -> Tuple[str, str, float, Optional[str], Optional[str]]:
    """Extract core fields from a pytest test dict.

    Args:
      t: Pytest test dict with keys 'nodeid', 'outcome', and optional 'call'.

    Returns:
      Tuple (nodeid, outcome, duration, longrepr, message), where:
        - longrepr: skip reason (tuple-string normalized if needed), truncated.
        - message: crash message with normalized spaces, truncated (or None).
    """
    nodeid = t["nodeid"]
    outcome = t["outcome"]
    call = t.get("call") or {}
    duration = float(call.get("duration", 0.0))

    longrepr_raw = call.get("longrepr")
    if (
        isinstance(longrepr_raw, str)
        and longrepr_raw.startswith("(")
        and longrepr_raw.endswith(")")
    ):
        longrepr_raw = extract_skip_reason(longrepr_raw)
    longrepr = shorten(longrepr_raw)

    message = None
    crash = call.get("crash")
    if isinstance(crash, dict):
        # Normalize excessive/irregular whitespace, then truncate.
        message = shorten(" ".join(str(crash.get("message", "")).split())) or None

    return nodeid, outcome, duration, longrepr, message


# -----------------------------
# Skip reason categorizer
# -----------------------------
@lru_cache(maxsize=4096)
def categorize_reason(reason: Optional[str]) -> str:
    """Map a skip/crash reason to a category label.

    Matching is case- and whitespace-insensitive. Rules are evaluated in order;
    the first match wins. Unknown/empty reasons fall back to DEFAULT_LABEL.
    """
    if not reason:
        return DEFAULT_LABEL

    s = " ".join(str(reason).split()).casefold()

    # Lazily compile rules once and cache them on the function object.
    try:
        rules = categorize_reason._RULES  # type: ignore[attr-defined]
    except AttributeError:
        rules_raw = [
            # ROCm
            {
                "any": ["skip on rocm", "skip for rocm"],
                "label": "Skipped on ROCm",
            },
            # Mosaic
            {"contains": "mosaic", "label": "Mosaic"},
            # Memory
            {
                "contains": "memory size limit exceeded",
                "label": "Memory Limit Exceeded",
            },
            # Device constraints / applicability
            {
                "any": [
                    "tpu",
                    "x64",
                    "x32",
                    "backend is not cpu",
                    "only for cpu",
                    "at least",
                ],
                "label": "Device Inapplicability",
            },
            {
                "contains": "memories do not work on cpu and gpu backends yet",
                "label": "Device Inapplicability",
            },
            {
                "contains": "test enabled only for cpu",
                "label": "Device Inapplicability",
            },
            {
                "contains": "jax implements eig only on cpu",
                "label": "Device Inapplicability",
            },
            {
                "contains": "schur decomposition is only implemented on cpu",
                "label": "Device Inapplicability",
            },
            {"all": ["test", "requires", "device"], "label": "Device Inapplicability"},
            # Missing module / API / plugin
            {
                "any": ["cuda", "sm90", "sm80", "cudnn", "magma is not installed"],
                "label": "Missing Module/API/Plugin",
            },
            {"contains": "no module named", "label": "Missing Module/API/Plugin"},
            {
                "regex": r"tests?\s+require(?:s)?\s+.+\s+plugin",
                "label": "Missing Module/API/Plugin",
            },
            {"contains": "requires", "label": "Missing Module/API/Plugin"},
            # Upstream / generic
            {"contains": "not supported on device", "label": "Skipped Upstream"},
            {
                "contains": "skipping big tests under sanitizers due to slowdown",
                "label": "Skipped Upstream",
            },
            {"contains": "not implemented", "label": "Skipped Upstream"},
            {"contains": "not relevant", "label": "Skipped Upstream"},
            {"contains": "dimension", "label": "Skipped Upstream"},
            {"contains": "support", "label": "Skipped Upstream"},
        ]
        compiled = []
        for r in rules_raw:
            r = dict(r)
            if "regex" in r:
                r["regex"] = re.compile(r["regex"])
            compiled.append(r)
        categorize_reason._RULES = tuple(compiled)  # type: ignore[attr-defined]
        rules = categorize_reason._RULES

    for r in rules:
        if "contains" in r and r["contains"] in s:
            return r["label"]
        if "any" in r and any(k in s for k in r["any"]):
            return r["label"]
        if "all" in r and all(k in s for k in r["all"]):
            return r["label"]
        if "regex" in r and r["regex"].search(s):
            return r["label"]
    return DEFAULT_LABEL


# -----------------------------
# DB Ops
# -----------------------------
def connect():
    """Open a MySQL connection from environment variables."""
    return mysql.connector.connect(
        host=os.environ["ROCM_JAX_DB_HOSTNAME"],
        user=os.environ["ROCM_JAX_DB_USERNAME"],
        password=os.environ["ROCM_JAX_DB_PASSWORD"],
        database=os.environ["ROCM_JAX_DB_NAME"],
        autocommit=False,
    )


def insert_run(cur, created_at: datetime, meta: dict) -> int:
    """Insert one row into ci_runs and return run_id. Idempotence is not enforced here."""
    cur.execute(
        """
       INSERT INTO ci_runs (
           created_at, commit_sha, runner_label, ubuntu_version,
           rocm_version, build_num, github_run_id, run_tag, logs_path
       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
       """,
        (
            created_at,
            meta["commit_sha"],
            meta["runner_label"],
            meta["ubuntu_version"],
            meta["rocm_version"],
            meta["build_num"],
            meta["github_run_id"],
            meta["run_tag"],
            meta["logs_dir"],
        ),
    )
    return cur.lastrowid


def sync_tests_and_get_ids(cur, tests: List[dict]) -> Dict[Tuple[str, str, str], int]:
    """Ensure all tests exist in ci_tests and return an ID mapping.

    Uses a TEMPORARY TABLE for efficiency with large runs:
      1) Bulk insert unique (filename, classname, test_name) into a temp table.
      2) INSERT any missing rows into ci_tests in one set operation.
      3) SELECT back (file, class, test) -> id mapping in one query.
    """
    uniq = {nodeid_parts(t["nodeid"]) for t in tests}
    if not uniq:
        return {}

    cur.execute(
        """
       CREATE TEMPORARY TABLE tmp_tests (
         filename  VARCHAR(100) NOT NULL,
         classname VARCHAR(100) NOT NULL,
         test_name VARCHAR(500) NOT NULL,
         PRIMARY KEY (filename, classname, test_name)
       ) ENGINE=InnoDB
       """
    )
    cur.executemany(
        "INSERT IGNORE INTO tmp_tests (filename, classname, test_name) VALUES (%s,%s,%s)",
        list(uniq),
    )

    cur.execute(
        """
       INSERT INTO ci_tests (filename, classname, test_name)
       SELECT s.filename, s.classname, s.test_name
       FROM tmp_tests s
       LEFT JOIN ci_tests t
         ON t.filename = s.filename
        AND t.classname = s.classname
        AND t.test_name = s.test_name
       WHERE t.id IS NULL
       """
    )

    cur.execute(
        """
       SELECT t.id, s.filename, s.classname, s.test_name
       FROM tmp_tests s
       JOIN ci_tests t
         ON t.filename = s.filename
        AND t.classname = s.classname
        AND t.test_name = s.test_name
       """
    )
    mapping: Dict[Tuple[str, str, str], int] = {}
    for test_id, f, c, n in cur.fetchall():
        mapping[(f, c, n)] = test_id
    return mapping


def batch_insert_results(
    cur,
    rows: List[
        Tuple[int, int, str, float, Optional[str], Optional[str], Optional[str]]
    ],
) -> None:
    """Bulk insert/update result rows in chunks.

    Uses ON DUPLICATE KEY UPDATE to keep results idempotent per (run_id, test_id).
    """
    if not rows:
        return

    sql = """
       INSERT INTO ci_results
           (run_id, test_id, outcome, duration, longrepr, message, skip_label)
       VALUES (%s,%s,%s,%s,%s,%s,%s)
       ON DUPLICATE KEY UPDATE
           outcome=VALUES(outcome),
           duration=VALUES(duration),
           longrepr=VALUES(longrepr),
           message=VALUES(message),
           skip_label=VALUES(skip_label)
    """
    for i in range(0, len(rows), BATCH_SIZE):
        cur.executemany(sql, rows[i : i + BATCH_SIZE])


# -----------------------------
# Entry point
# -----------------------------
def upload_pytest_results(
    logs_dir: Path,
    *,
    runner_label: str,
    ubuntu_version: str,
    rocm_version: str,
    commit_sha: str,
    build_num: str,
    github_run_id: str,
    run_tag: str,
) -> None:
    """Load per-test JSON reports and upload results to MySQL.

    Flow:
      1) Parse JSONs and gather tests.
      2) Insert a ci_runs row and get run_id.
      3) Ensure all tests exist; get (file,class,test) -> test_id map.
      4) Bulk insert/update ci_results for this run.
    """
    files = list_test_jsons(logs_dir)
    if not files:
        print(f"No JSON files under {logs_dir}")
        raise SystemExit(2)  # input error

    created_at, tests = load_from_per_test_jsons(files)
    if not tests:
        print("No tests found in JSONs")
        raise SystemExit(2)  # input error

    conn = connect()
    cur = conn.cursor()
    try:
        run_id = insert_run(
            cur,
            created_at,
            {
                "runner_label": runner_label,
                "ubuntu_version": ubuntu_version,
                "rocm_version": rocm_version,
                "commit_sha": commit_sha,
                # force cast numeric-like fields to match DB INT/BIGINT if possible
                "build_num": int(build_num) if str(build_num).isdigit() else build_num,
                "github_run_id": (
                    int(github_run_id)
                    if str(github_run_id).isdigit()
                    else github_run_id
                ),
                "run_tag": run_tag,
                "logs_dir": str(logs_dir),
            },
        )

        test_id_map = sync_tests_and_get_ids(cur, tests)

        rows: List[
            Tuple[int, int, str, float, Optional[str], Optional[str], Optional[str]]
        ] = []
        append = rows.append
        for t in tests:
            nodeid, outcome, duration, longrepr, message = extract_result_fields(t)
            f, c, n = nodeid_parts(nodeid)
            test_id = test_id_map[(f, c, n)]
            skip_label = categorize_reason(longrepr) if outcome == "skipped" else None
            append(
                (
                    run_id,
                    test_id,
                    outcome,
                    duration,
                    longrepr,
                    message,
                    skip_label,
                )
            )
        batch_insert_results(cur, rows)
        conn.commit()
        print(
            f"[summary] run_id={run_id} total_results={len(rows)} unique_tests={len(test_id_map)}"
        )
    except MySQLError as e:
        conn.rollback()
        print(f"MySQL error: {e}")
        raise SystemExit(10)  # db error
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload pytest JSON reports to MySQL")
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--ubuntu-version", required=True)
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--build-num", required=True)
    parser.add_argument("--github-run-id", required=True)

    args = parser.parse_args()

    upload_pytest_results(
        Path(args.logs_dir),
        run_tag=args.run_tag,
        runner_label=args.runner_label,
        ubuntu_version=args.ubuntu_version,
        rocm_version=args.rocm_version,
        commit_sha=args.commit_sha,
        build_num=args.build_num,
        github_run_id=args.github_run_id,
    )
