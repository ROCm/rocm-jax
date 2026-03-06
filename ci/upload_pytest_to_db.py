#!/usr/bin/env python3
"""
Ingest JAX Pytest results from S3 into MySQL.

Recursively locates one Pytest report under given log dir

Tables:
 - jax_ci_runs:    one row per run
 - jax_ci_tests:   one row per unique test
 - jax_ci_results: one row per test per run

Run-level manifest (GitHub vars, etc.) is sourced from the CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# pylint: disable=import-error
import mysql.connector
from mysql.connector import Error as MySQLError

# -----------------------------
# Constants
# -----------------------------
TEXT_LIMIT = 250
BATCH_SIZE = 2000
DEFAULT_LABEL = "Skipped Upstream"
MANIFEST_FILENAME = "run-manifest.json"


# -----------------------------
# Helpers
# -----------------------------
def extract_skip_reason(reason: str) -> str:
    """Parse pytest skip longrepr tuple-string into its reason text.

    Example input: "('/path/test_x.py', 42, 'Skipped: some reason')"
    Also works for xdist header:
      "[gw0] ... \\n('/path/test_x.py', 42, 'Skipped: some reason')"
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


def pipe_split(raw: Optional[str]) -> List[str]:
    """Split a pipe-separated manifest field into a list of values."""
    if not raw:
        return []
    return [p for p in (x.strip() for x in raw.split("|")) if p]


def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO8601 timestamp into UTC datetime."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def parse_run_key_and_combo(artifact_uri: str) -> tuple[str, str]:
    """Extract (run_key, combo) from the artifact URI path.

    Expected format: .../<run_key>/<combo>
    """
    parts = [p for p in artifact_uri.split("/") if p]
    if len(parts) < 2:
        raise SystemExit(f"Invalid artifact_uri format: {artifact_uri}")
    return parts[-2], parts[-1]  # run_key, combo


# -----------------------------
# Input/File Loading
# -----------------------------
def find_pytest_report_json(local_logs_dir: Path) -> Optional[Path]:
    """Find the single Pytest JSON report under the given logs directory, if present"""
    ignore = {MANIFEST_FILENAME, "last_running.json"}
    reports = [
        p
        for p in local_logs_dir.rglob("*.json")
        if p.is_file()
        and p.name not in ignore
        and not p.name.endswith("last_running.json")
    ]
    if not reports:
        return None
    if len(reports) != 1:
        listing = "\n  - " + "\n  - ".join(
            str(p.relative_to(local_logs_dir)) for p in sorted(reports)
        )
        raise SystemExit(
            f"Expected exactly ONE pytest JSON report; found {len(reports)}:{listing}"
        )
    return reports[0]


def load_from_pytest_json(path: Path) -> Tuple[Optional[datetime], List[dict]]:
    """Load pytest JSON report and return (report_created_at, tests), if present"""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        tests = data.get("tests", [])
        created = data.get("created")
        report_created_at = (
            datetime.fromtimestamp(float(created), tz=timezone.utc).replace(tzinfo=None)
            if created is not None
            else None
        )
        return report_created_at, tests
    if isinstance(data, list):
        return None, data
    raise ValueError(f"Unexpected report JSON structure: {path}")


def load_manifest(local_logs_dir: Path) -> dict:
    """Load CI run metadata from run-manifest.json.

    Schema Version_1 Fields:
      run_started_at      CI run start time
      run_completed_at    CI run completion time
      github_run_url      URL of the GitHub Actions run
      github_repository   Repository name (e.g. "ROCm/jax")
      github_ref_name     Branch or tag name
      github_ref          Full Git reference
      github_sha          Commit SHA for the run
      github_event_name   GitHub event type (push, pull_request, etc.)
      github_run_id       GitHub Actions run identifier
      github_run_attempt  Retry attempt number
      github_run_number   Sequential run number
      github_workflow     Workflow name
      github_job          Job name within the workflow
      python_version      Python version used in the run
      rocm_version        ROCm version used
      rocm_tag            ROCm container tag
      isnighty            Whether it is a nightly or continuous run
      gpu_count           Number of GPUs used
      runner              CI runner label
      base_image_name     Base container image name
      base_image_digest   Container image digest
      jax_packages_raw    Installed JAX Python packages (pipe-separated)
      wheels_sha_raw      Wheel SHA256 list (pipe-separated)

    Some fields may be optional and missing."""
    p = local_logs_dir / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"{MANIFEST_FILENAME} not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# -----------------------------
# Run Fields Preparation
# -----------------------------
def require_field(m: dict, key: str):
    """Return a required manifest field, failing early if it is missing."""
    v = m.get(key)
    if v in (None, ""):
        raise SystemExit(f"Manifest missing required field: {key}")
    return v


def packages_json_and_jax_version(
    raw: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Parse package list into JSON and extract the JAX version."""
    if not raw:
        return None, None
    pkgs = []
    jax_ver = None
    for item in pipe_split(raw):
        name, sep, ver = item.partition("==")
        name = name.strip()
        ver = ver.strip() if sep else None
        pkgs.append({"name": name, "version": ver, "raw": item})
        if name == "jax" and ver:
            jax_ver = ver
    return json.dumps(pkgs), jax_ver


def wheels_json(raw: Optional[str]) -> Optional[str]:
    """Parse wheels metadata into normalized JSON."""
    if not raw:
        return None
    wheels = []
    for line in pipe_split(raw):
        m = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line)
        if m:
            wheels.append({"sha256": m.group(1).lower(), "file": m.group(2).strip()})
        else:
            wheels.append({"sha256": None, "file": line})
    return json.dumps(wheels)


def build_run_fields(  # pylint: disable=too-many-locals
    m: dict,
    *,
    artifact_uri: str,
    run_tag: str,
    gpu_tag: str,
) -> dict:
    """Normalize manifest fields for jax_ci_runs insertion."""
    run_key, combo = parse_run_key_and_combo(artifact_uri)
    github_repository = require_field(m, "github_repository")
    github_ref_name = require_field(m, "github_ref_name")
    github_run_id = int(require_field(m, "github_run_id"))
    python_version = require_field(m, "python_version")
    rocm_version = require_field(m, "rocm_version")
    runner = require_field(m, "runner")
    is_nightly = require_field(m, "is_nightly")
    if is_nightly not in {"nightly", "continuous"}:
        raise SystemExit(f"Invalid is_nightly value: {is_nightly}")
    github_run_attempt = int(m.get("github_run_attempt") or 1)
    github_run_number = (
        int(m["github_run_number"])
        if m.get("github_run_number") not in (None, "")
        else None
    )
    pkgs_json, jax_ver_guess = packages_json_and_jax_version(m.get("jax_packages_raw"))
    whl_json = wheels_json(m.get("wheels_sha_raw"))
    return {
        "github_repository": github_repository,
        "github_ref_name": github_ref_name,
        "github_ref": m.get("github_ref"),
        "github_event_name": m.get("github_event_name"),
        "github_run_url": m.get("github_run_url"),
        "github_sha": m.get("github_sha"),
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
        "github_run_number": github_run_number,
        "github_workflow": m.get("github_workflow"),
        "github_job": m.get("github_job"),
        "runner": runner,
        "python_version": python_version,
        "rocm_version": rocm_version,
        "rocm_tag": m.get("rocm_tag"),
        "gpu_count": m.get("gpu_count"),
        "gpu_tag": gpu_tag,
        "is_nightly": is_nightly,
        "run_tag": run_tag,
        "run_key": run_key,
        "combo": combo,
        "artifact_uri": artifact_uri,
        "jax_version": m.get("jax_version") or jax_ver_guess,
        "jax_commit": m.get("jax_commit"),
        "xla_commit": m.get("xla_commit"),
        "base_image_name": m.get("base_image_name"),
        "base_image_digest": m.get("base_image_digest"),
        "packages_json": pkgs_json,
        "wheels_json": whl_json,
        "run_started_at": parse_iso_dt(m.get("run_started_at")),
        "run_completed_at": parse_iso_dt(m.get("run_completed_at")),
    }


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
    if isinstance(longrepr_raw, str) and longrepr_raw:
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


def find_existing_run_id(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    cur,
    github_repository: str,
    github_ref_name: str,
    is_nightly: str,
    run_key: str,
    combo: str,
) -> Optional[int]:
    """Return run id for an existing logical run, if present."""
    cur.execute(
        """
       SELECT id
       FROM jax_ci_runs
       WHERE github_repository = %s
         AND github_ref_name = %s
         AND is_nightly = %s
         AND run_key = %s
         AND combo = %s
       LIMIT 1
       """,
        (github_repository, github_ref_name, is_nightly, run_key, combo),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def insert_run(cur, report_created_at: Optional[datetime], fields: dict) -> int:
    """Insert one row into jax_ci_runs and return run_id.

    Runs are treated as immutable; duplicate logical runs should be
    detected before calling this function."""
    ingested_at = datetime.now(timezone.utc).replace(tzinfo=None)
    if report_created_at is None:
        report_created_at = ingested_at
    else:
        report_created_at = report_created_at.replace(tzinfo=None)

    cur.execute(
        """
       INSERT INTO jax_ci_runs (
         report_created_at, run_started_at, run_completed_at, ingested_at,
         github_repository, github_ref_name, github_ref, github_event_name,
         github_run_url, github_sha,
         github_run_id, github_run_attempt, github_run_number,
         github_workflow, github_job,
         runner, python_version, rocm_version, rocm_tag,
         gpu_count, gpu_tag, is_nightly,
         run_tag, run_key, combo, artifact_uri,
         jax_version, jax_commit, xla_commit,
         base_image_name, base_image_digest,
         packages_json, wheels_json
       ) VALUES (
         %s, %s, %s, %s,
         %s, %s, %s, %s,
         %s, %s,
         %s, %s, %s,
         %s, %s,
         %s, %s, %s, %s,
         %s, %s, %s,
         %s, %s, %s, %s,
         %s, %s, %s,
         %s, %s,
         %s, %s
       )
       """,
        (
            report_created_at,
            fields["run_started_at"],
            fields["run_completed_at"],
            ingested_at,
            fields["github_repository"],
            fields["github_ref_name"],
            fields["github_ref"],
            fields["github_event_name"],
            fields["github_run_url"],
            fields["github_sha"],
            fields["github_run_id"],
            fields["github_run_attempt"],
            fields["github_run_number"],
            fields["github_workflow"],
            fields["github_job"],
            fields["runner"],
            fields["python_version"],
            fields["rocm_version"],
            fields["rocm_tag"],
            fields["gpu_count"],
            fields["gpu_tag"],
            fields["is_nightly"],
            fields["run_tag"],
            fields["run_key"],
            fields["combo"],
            fields["artifact_uri"],
            fields["jax_version"],
            fields["jax_commit"],
            fields["xla_commit"],
            fields["base_image_name"],
            fields["base_image_digest"],
            fields["packages_json"],
            fields["wheels_json"],
        ),
    )
    return int(cur.lastrowid)


def sync_tests_and_get_ids(cur, tests: List[dict]) -> Dict[Tuple[str, str, str], int]:
    """Ensure all tests exist in jax_ci_tests and return an ID mapping.

    Uses a TEMPORARY TABLE for efficiency with large runs:
      1) Bulk insert unique (filename, classname, test_name) into a temp table.
      2) INSERT any missing rows into jax_ci_tests in one set operation.
      3) SELECT back (file, class, test) -> id mapping in one query.
    """
    uniq = {nodeid_parts(t["nodeid"]) for t in tests}
    if not uniq:
        return {}

    cur.execute("DROP TEMPORARY TABLE IF EXISTS tmp_tests_")
    # fmt: off
    cur.execute(
        """
       CREATE TEMPORARY TABLE tmp_tests_ (
         filename  VARCHAR(100) NOT NULL,
         classname VARCHAR(100) NOT NULL,
         test_name VARCHAR(500) NOT NULL,
         PRIMARY KEY (filename, classname, test_name)
       ) ENGINE=InnoDB
       """
    )
    cur.executemany(
        "INSERT IGNORE INTO tmp_tests_ (filename, classname, test_name) VALUES (%s,%s,%s)",
        list(uniq),
    )

    cur.execute(
        """
       INSERT INTO jax_ci_tests (filename, classname, test_name)
       SELECT s.filename, s.classname, s.test_name
       FROM tmp_tests_ s
       LEFT JOIN jax_ci_tests t
         ON t.filename = s.filename
        AND t.classname = s.classname
        AND t.test_name = s.test_name
       WHERE t.id IS NULL
       """
    )

    cur.execute(
        """
       SELECT t.id, s.filename, s.classname, s.test_name
       FROM tmp_tests_ s
       JOIN jax_ci_tests t
         ON t.filename = s.filename
        AND t.classname = s.classname
        AND t.test_name = s.test_name
       """
    )
    # fmt: on
    return {(f, c, n): int(test_id) for (test_id, f, c, n) in cur.fetchall()}


def batch_insert_results(cur, rows) -> None:
    """Bulk insert/update result rows in chunks.

    Uses ON DUPLICATE KEY UPDATE to keep results idempotent per (run_id, test_id).
    """
    if not rows:
        return

    sql = """
       INSERT INTO jax_ci_results
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
def upload_pytest_results(  # pylint: disable=too-many-locals
    local_logs_dir: Path,
    *,
    run_tag: str,
    gpu_tag: str,
    artifact_uri: str,
) -> None:
    """Load per-test JSON reports and upload results to MySQL.

    Flow:
      1) Parse JSONs and gather tests.
      2) Insert a jax_ci_runs row and get run_id.
      3) Ensure all tests exist; get (file,class,test) -> test_id map.
      4) Bulk insert/update jax_ci_results for this run.
    """
    report = find_pytest_report_json(local_logs_dir)
    if report is None:
        report_created_at = None
        tests = []
    else:
        report_created_at, tests = load_from_pytest_json(report)

    manifest = load_manifest(local_logs_dir)
    fields = build_run_fields(
        manifest,
        artifact_uri=artifact_uri,
        run_tag=run_tag,
        gpu_tag=gpu_tag,
    )

    conn = connect()
    cur = conn.cursor()
    try:
        existing_run_id = find_existing_run_id(
            cur,
            fields["github_repository"],
            fields["github_ref_name"],
            fields["is_nightly"],
            fields["run_key"],
            fields["combo"],
        )
        if existing_run_id is not None:
            conn.rollback()
            print(
                "[DUPLICATE] run already exists: "
                f"run_id={existing_run_id} "
                f"repo={fields['github_repository']} "
                f"ref={fields['github_ref_name']} "
                f"is_nightly={fields['is_nightly']} "
                f"run_key={fields['run_key']} "
                f"combo={fields['combo']}"
            )
            return
        run_id = insert_run(cur, report_created_at, fields)

        rows = []
        test_id_map = {}

        if tests:
            test_id_map = sync_tests_and_get_ids(cur, tests)
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

                rows.append(
                    (run_id, test_id, outcome, duration, longrepr, message, skip_label)
                )
            batch_insert_results(cur, rows)
        conn.commit()
        print(
            f"[summary] run_id={run_id} total_results={len(rows)} unique_tests={len(test_id_map)}"
        )
        # NOTE: optionally print Grafana dashboard URL, e.g. {URL}?var-run_id={id}
    except MySQLError as e:
        conn.rollback()
        # INSERT may still hit a duplicate key (e.g. artifact_uri or logical identity)
        # even if the earlier SELECT did not detect it.
        if getattr(e, "errno", None) == 1062:
            print(
                "[DUPLICATE] insert hit unique constraint: "
                f"repo={fields['github_repository']} "
                f"ref={fields['github_ref_name']} "
                f"is_nightly={fields['is_nightly']} "
                f"run_key={fields['run_key']} "
                f"combo={fields['combo']} "
                f"artifact_uri={fields['artifact_uri']}"
            )
            return
        raise SystemExit(f"MySQL error: {e}") from e
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pytest DB uploader."""
    p = argparse.ArgumentParser(description="Upload pytest report + manifest to MySQL")
    p.add_argument("--local_logs_dir", required=True, help="Directory with JSON files")
    p.add_argument("--run-tag", required=True, help="Run tag, e.g. ci-run")
    p.add_argument("--gpu-tag", required=True, help="GPU architecture, e.g. MI350")
    p.add_argument("--artifact_uri", required=True, help="Unique artifact path for CI")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_pytest_results(
        Path(args.local_logs_dir),
        run_tag=args.run_tag,
        gpu_tag=args.gpu_tag,
        artifact_uri=args.artifact_uri,
    )
