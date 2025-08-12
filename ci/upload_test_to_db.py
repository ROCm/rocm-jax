"""Upload pytest reports to DB"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# pylint: disable=import-error
import mysql.connector
from mysql.connector import Error as MySQLError


def load_pytest_json(filepath: str) -> dict:
    """Read a pytest JSON report file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def connect_to_test_db():
    """Open MySQL connection from env vars; autocommit enabled"""
    try:
        return mysql.connector.connect(
            host=os.environ["ROCM_JAX_DB_HOSTNAME"],
            user=os.environ["ROCM_JAX_DB_USERNAME"],
            password=os.environ["ROCM_JAX_DB_PASSWORD"],
            database=os.environ["ROCM_JAX_DB_NAME"],
            autocommit=True,  # commit changes automatically
        )
    except MySQLError as e:
        raise RuntimeError(f"MySQL connection failed: {e}") from e


def insert_test_run(cursor, report: dict, metadata: dict) -> int:
    """Insert a test run row and return its id"""
    summary = report.get("summary", {})

    query = """
        INSERT INTO ci_test_runs (
            runner_label, ubuntu_version, rocm_version,
            logs_dir, github_run_id, commit_sha,
            created_at, total, passed, failed, skipped
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    values = (
        metadata["runner_label"],
        metadata["ubuntu_version"],
        metadata["rocm_version"],
        metadata["logs_dir"],
        metadata["github_run_id"],
        metadata["commit_sha"],
        datetime.fromtimestamp(report["created"], timezone.utc),
        summary.get("total", 0),
        summary.get("passed", 0),
        summary.get("failed", 0),
        summary.get("skipped", 0),
    )
    cursor.execute(query, values)
    return cursor.lastrowid


def insert_test_cases(cursor, run_id: int, tests: list) -> None:
    """Insert all test cases for a given run id"""
    query = """
        INSERT INTO ci_test_cases (
            run_id, nodeid, outcome, duration, longrepr, message
        ) VALUES (%s, %s, %s, %s, %s, %s);
    """
    for test in tests:
        call = test.get("call", {}) or {}
        crash = call.get("crash", {}) or {}
        values = (
            run_id,
            test.get("nodeid"),
            test.get("outcome"),
            float(call.get("duration", 0.0)),
            (call.get("longrepr", "") or "")[:1000],  # useful when skipped
            (crash.get("message", "") or "")[:1000],  # useful when failed
        )
        cursor.execute(query, values)


# Spefically needed for crashed/aborted tests:
# when a test crashes or aborts, pytest fails to write the JSON report, but HTML exists.
# So, collect set of test_names with regex *.json/*.html in logs_dir (-exclude non-test files)
# Then, in main() if JSON missing, fallback to HTML and create placeholder with summary=-1 values
SKIP_FILES = {
    "final_compiled_report.json",
    "final_compiled_report.html",
    "collect_module_log.jsonl",
}
NAME_RX = re.compile(
    r"^(?P<name>.+?)\.(?P<ext>json|html)$"
)  # *.(json|html) -> test_name


def collect_test_names(logs_dir: str) -> set[str]:
    "Collect unique test names from logs_dir, excluding SKIP_FILES"
    names = set()
    p = Path(logs_dir)
    for f in p.iterdir():
        if not f.is_file():
            continue
        if f.name in SKIP_FILES:
            continue
        m = NAME_RX.match(f.name)
        if m:
            names.add(m.group("name"))
    return names


def main():
    """Main entrypoint: parse args, load logs, write to DB"""
    parser = argparse.ArgumentParser(
        description="Upload pytest JSON report to database"
    )

    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--ubuntu-version", required=True)
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--commit-sha", required=True)

    args = parser.parse_args()

    conn = connect_to_test_db()
    cursor = conn.cursor()

    try:
        all_test_names = collect_test_names(args.logs_dir)
        last_json_created = None  # timestamp from last JSON; used in fallback
        for test_name in sorted(all_test_names):
            json_path = os.path.join(args.logs_dir, f"{test_name}.json")

            if os.path.exists(json_path):
                try:
                    report = load_pytest_json(json_path)
                    last_json_created = report["created"]
                    run_id = insert_test_run(cursor, report, vars(args))
                    insert_test_cases(cursor, run_id, report.get("tests", []))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"{test_name} could not proceed: {e}")
            else:
                # Fallback when JSON missing for crashed tests
                try:
                    report = {
                        "created": last_json_created
                        or datetime.now(timezone.utc).timestamp(),
                        "summary": {
                            "total": -1,
                            "passed": -1,
                            "failed": -1,
                            "skipped": -1,
                        },
                        "tests": [
                            {
                                "nodeid": test_name + ".py",
                                "outcome": "unknown",
                                "call": {"duration": -1},
                            }
                        ],
                    }
                    run_id = insert_test_run(cursor, report, vars(args))
                    insert_test_cases(cursor, run_id, report.get("tests", []))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"{test_name} could not proceed: {e}")
        conn.commit()
    except MySQLError as e:
        conn.rollback()
        raise RuntimeError(f"MySQL error: {e}") from e
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
