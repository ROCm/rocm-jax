#!/usr/bin/env python3
"""
Upload pytest results to MySQL.
Tables:
 - pytest_ci_runs:    one row per run
 - pytest_ci_tests:   one row per unique test
 - pytest_ci_results: one row per test per run
"""

from __future__ import annotations

import argparse
import os
import re
import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# pylint: disable=import-error
import mysql.connector
from mysql.connector import Error as MySQLError

# -----------------------------
# Constants
# -----------------------------
TEXT_LIMIT = 250
BATCH_SIZE = 2000
DEFAULT_LABEL = "Skipped Upstream"


# -----------------------------
# Helpers
# -----------------------------
def extract_skip_reason(reason: str) -> str:
    """Parse pytest skip longrepr tuple-string into its reason text.

    Example input: "('/path/test_x.py', 42, 'Skipped: some reason')"
    """

    # strip outer parentheses,
    # then split into 3 parts
    parts = reason[1:-1].split(",", 2)
    if len(parts) != 3:
        return reason

    msg = parts[2].strip()
    # drop matching quotes if any
    if msg[:1] in {"'", '"'} and msg[-1:] == msg[:1]:
        msg = msg[1:-1]
    return msg


def nodeid_parts(nodeid: str) -> Tuple[str, str, str]:
    """Split pytest nodeid into (filename, classname, test_name).

    Support both variants: "file::Class::test" and "file::test" in nodeids.
    """
    parts = nodeid.split("::", 2)
    f = parts[0]
    if len(parts) == 2:
        return f, "", parts[1]
    c = parts[1]
    t = parts[2]
    return f, c, t


def load_metadata(logs_dir: Path) -> dict:
    """Load metadata.json from logs_dir, which should contain run-level info like commit, runner"""
    meta_path = logs_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {logs_dir}")
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_single_report_json(logs_dir: Path) -> Path:
    """Return the single pytest JSON report in logs_dir (ignores metadata.json)."""
    jsons = [
        p
        for p in logs_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".json"
        and p.name not in {"metadata.json"}
        and not p.name.endswith("last_running.json")
    ]
    if not jsons:
        print(f"No JSON report found in {logs_dir}")
        raise SystemExit(2)
    if len(jsons) != 1:
        print(
            f"Expected exactly ONE JSON report, found {len(jsons)}: "
            + ", ".join(p.name for p in jsons)
        )
        raise SystemExit(2)
    return jsons[0]


def load_from_single_json(path: Path) -> Tuple[datetime, List[dict]]:
    """Load a pytest JSON report and return (created_at, tests)."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        tests = data.get("tests", [])
        created = data.get("created")
        if created is not None:
            created_at = datetime.fromtimestamp(float(created))
        else:
            created_at = datetime.fromtimestamp(path.stat().st_mtime)
        return created_at, tests
    if isinstance(data, list):
        created_at = datetime.fromtimestamp(path.stat().st_mtime)
        return created_at, data
    raise ValueError(f"Unexpected JSON structure in {path.name}")


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
    longrepr = str(longrepr_raw)[:TEXT_LIMIT] if longrepr_raw is not None else None

    message = None
    crash = call.get("crash")
    if isinstance(crash, dict):
        # Normalize excessive/irregular whitespace, then truncate.
        raw_msg = crash.get("message", "")
        msg = " ".join(str(raw_msg).split())
        message = msg[:TEXT_LIMIT] if msg else None

    return nodeid, outcome, duration, longrepr, message


# -----------------------------
# Skip reason categorizer
# -----------------------------
# Precompile skip categorization rules (regex etc.) once.
# categorize_reason() reuses them; lru_cache avoids recompute.
# Rules are evaluated in order - more specific rules should come before generic ones
_RULES_RAW = [
    # TPU-specific (checked first)
    {"contains": "tpu", "label": "TPU-Only"},
    # Mosaic (check reason only, filename/testname checked separately)
    {"contains": "mosaic", "label": "Mosaic"},
    # ROCm-specific checks
    {"any": ["skip on rocm", "skip for rocm"], "label": "Not Supported on ROCm"},
    {"all": ["not supported on", "rocm"], "label": "Not Supported on ROCm"},
    {"contains": "is not available for rocm", "label": "Not Supported on ROCm"},
    # Multiple devices required (before generic "support" check)
    {"all": [">=", "devices"], "label": "Multiple Devices Required"},
    {"all": ["test", "requires", "device"], "label": "Multiple Devices Required"},
    # NVIDIA-specific
    {
        "any": [
            "cuda",
            "sm90",
            "sm100a",
            "sm80",
            "cudnn",
            "nvidia",
            "cupy",
            "capability",
        ],
        "label": "NVIDIA-Specific",
    },
    {"contains": "at least", "label": "NVIDIA-Specific"},
    # Apple-specific
    {"any": ["metal", "apple"], "label": "Apple-Specific"},
    # CPU-only tests
    {"contains": "test enabled only for cpu", "label": "CPU-Only"},
    {"contains": "jax implements eig only on cpu", "label": "CPU-Only"},
    {"contains": "schur decomposition is only implemented on cpu", "label": "CPU-Only"},
    {"contains": "backend is not cpu", "label": "CPU-Only"},
    {"contains": "only for cpu", "label": "CPU-Only"},
    # Device inapplicability
    {"contains": "x64", "label": "Device Inapplicability"},
    {"contains": "x32", "label": "Device Inapplicability"},
    {
        "contains": "memories do not work on cpu and gpu backends yet",
        "label": "Device Inapplicability",
    },
    # Missing modules/plugins
    {"contains": "magma is not installed", "label": "Missing Module/API/Plugin"},
    {"contains": "no module named", "label": "Missing Module/API/Plugin"},
    {"contains": "requires pytorch", "label": "Missing Module/API/Plugin"},
    {"contains": "requires tensorflow", "label": "Missing Module/API/Plugin"},
    {
        "regex": re.compile(r"tests?\s+require?\s+(.+?)\s+plugin", re.I),
        "label": "Missing Module/API/Plugin",
    },
    # Memory Limit
    {"contains": "memory size limit exceeded", "label": "Memory Limit Exceeded"},
    # Performance-related skips
    {"contains": "too slow", "label": "Too Slow (Skipped Upstream)"},
    {
        "contains": "skipping big tests under sanitizers due to slowdown",
        "label": "Too Slow (Skipped Upstream)",
    },
    # Maintenance-related skips
    {
        "any": ["unmaintained", "not maintained"],
        "label": "Currently Unmaintained (Skipped Upstream)",
    },
    # Generic "Skipped Upstream" checks (at the bottom)
    {"contains": "dimension", "label": "Skipped Upstream"},
    {"contains": "not supported in interpret mode", "label": "Skipped Upstream"},
    {"contains": "not implemented", "label": "Skipped Upstream"},
    {"contains": "not relevant", "label": "Skipped Upstream"},
    {"contains": "support", "label": "Skipped Upstream"},
]

_CATEG_RULES = tuple(dict(r) for r in _RULES_RAW)


@lru_cache(maxsize=4096)
def categorize_reason(reason: Optional[str]) -> str:
    """Map a skip reason to a category label.

    Matching is case- and whitespace-insensitive. Rules are evaluated in order;
    the first match wins. Unknown/empty reasons fall back to DEFAULT_LABEL.
    """
    if not reason:
        return DEFAULT_LABEL

    s = " ".join(str(reason).split()).casefold()

    for rule in _CATEG_RULES:
        if "contains" in rule and rule["contains"] in s:
            return rule["label"]
        if "any" in rule and any(k in s for k in rule["any"]):
            return rule["label"]
        if "all" in rule and all(k in s for k in rule["all"]):
            return rule["label"]
        if "regex" in rule and rule["regex"].search(s):
            return rule["label"]
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
    """Insert one row into pytest_ci_runs and return run_id. Idempotence is not enforced here."""
    cur.execute(
        """
       INSERT INTO pytest_ci_runs (
           created_at, commit_sha, runner_label, python_version, rocm_version,
           build_num, github_run_id, branch_name, repo_name, run_tag, logs_path
       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
       """,
        (
            created_at,
            meta["commit_sha"],
            meta["runner_label"],
            meta["python_version"],
            meta["rocm_version"],
            meta["build_num"],
            meta["github_run_id"],
            meta["branch_name"],
            meta["repo_name"],
            meta["run_tag"],
            meta["logs_dir"],
        ),
    )
    return cur.lastrowid


def sync_tests_and_get_ids(cur, tests: List[dict]) -> Dict[Tuple[str, str, str], int]:
    """Ensure all tests exist in pytest_ci_tests and return an ID mapping.

    Uses a TEMPORARY TABLE for efficiency with large runs:
      1) Bulk insert unique (filename, classname, test_name) into a temp table.
      2) INSERT any missing rows into pytest_ci_tests in one set operation.
      3) SELECT back (file, class, test) -> id mapping in one query.
    """
    uniq = {nodeid_parts(t["nodeid"]) for t in tests}
    if not uniq:
        return {}
    # fmt: off
    cur.execute(
        """
       CREATE TEMPORARY TABLE tmp_pytest_tests (
         filename  VARCHAR(100) NOT NULL,
         classname VARCHAR(100) NOT NULL,
         test_name VARCHAR(500) NOT NULL,
         PRIMARY KEY (filename, classname, test_name)
       ) ENGINE=InnoDB
       """
    )
    cur.executemany(
        "INSERT IGNORE INTO tmp_pytest_tests (filename, classname, test_name) VALUES (%s,%s,%s)",
        list(uniq),
    )

    cur.execute(
        """
       INSERT INTO pytest_ci_tests (filename, classname, test_name)
       SELECT s.filename, s.classname, s.test_name
       FROM tmp_pytest_tests s
       LEFT JOIN pytest_ci_tests t
         ON t.filename = s.filename
        AND t.classname = s.classname
        AND t.test_name = s.test_name
       WHERE t.id IS NULL
       """
    )

    cur.execute(
        """
       SELECT t.id, s.filename, s.classname, s.test_name
       FROM tmp_pytest_tests s
       JOIN pytest_ci_tests t
         ON t.filename = s.filename
        AND t.classname = s.classname
        AND t.test_name = s.test_name
       """
    )
    # fmt: on
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
       INSERT INTO pytest_ci_results
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
def upload_pytest_results(  # pylint: disable=too-many-arguments, too-many-locals
    logs_dir: Path,
    *,
    run_tag: str,
) -> None:
    """Load per-test JSON reports and upload results to MySQL.

    Flow:
      1) Parse JSONs and gather tests.
      2) Insert a pytest_ci_runs row and get run_id.
      3) Ensure all tests exist; get (file,class,test) -> test_id map.
      4) Bulk insert/update pytest_ci_results for this run.
    """
    report = find_single_report_json(logs_dir)
    created_at, tests = load_from_single_json(report)

    metadata = load_metadata(logs_dir)
    repo_name = metadata["repo_name"]
    branch_name = metadata["branch_name"]
    runner_label = metadata["runner_label"]
    python_version = str(metadata["python_version"]).replace(
        ".", ""
    )  # bash ${Py_version//.}
    rocm_version = str(metadata["rocm_version"]).replace(
        ".", ""
    )  # bash ${ROCM_VERSION//.}
    commit_sha = metadata["github_sha"] or metadata.get("commit_sha")
    github_run_id = int(metadata["github_run_id"])
    build_num = metadata.get("build_num")

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
                "python_version": python_version,
                "rocm_version": rocm_version,
                "commit_sha": commit_sha,
                "build_num": build_num,
                "github_run_id": github_run_id,
                "branch_name": branch_name,
                "repo_name": repo_name,
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

            # Categorize skip reason, with special check for Mosaic
            # in filename/testname (including mgpu)
            skip_label = None
            if outcome == "skipped":
                # Check if "mosaic" or "mgpu" is in filename or test name
                if (
                    "mosaic" in f.lower()
                    or "mosaic" in n.lower()
                    or "mgpu" in f.lower()
                ):
                    skip_label = "Mosaic"
                else:
                    skip_label = categorize_reason(longrepr)

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
        # NOTE: optionally print Grafana dashboard URL, e.g. {URL}?var-run_id={id}
    except MySQLError as e:
        conn.rollback()
        print(f"MySQL error: {e}")
        raise SystemExit(10) from e  # db error
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pytest DB uploader."""
    p = argparse.ArgumentParser(description="Upload pytest JSON reports to MySQL")
    p.add_argument("--logs_dir", required=True, help="Directory with JSON files")
    p.add_argument("--run-tag", required=True, help="Run label, e.g. ci-run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_pytest_results(
        Path(args.logs_dir),
        run_tag=args.run_tag,
    )
