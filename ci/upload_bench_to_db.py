#!/usr/bin/env python3
"""
Ingest ROCm benchmark result.json into MySQL.
Tables:
 - jax_ci_benchmark_runs
 - jax_ci_benchmark_metrics
 - jax_ci_benchmark_results
"""

from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import mysql.connector
from mysql.connector import Error as MySQLError

RESULT_FILE = "result.json"
BENCHMARK_RAW_FIELDS = (
    "requirements_raw",
    "workload_config_raw",
    "expected_config_raw",
    "benchmark_config_raw",
)


def connect():
    return mysql.connector.connect(
        host=os.environ["ROCM_JAX_DB_HOSTNAME"],
        user=os.environ["ROCM_JAX_DB_USERNAME"],
        password=os.environ["ROCM_JAX_DB_PASSWORD"],
        database=os.environ["ROCM_JAX_DB_NAME"],
        autocommit=False,
    )


def parse_iso_dt(value):
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
        tzinfo=None
    )


def norm(value) -> str:
    return str(value).strip().replace(".", "_").replace("-", "_")


def load_benchmark_result(log_dir: Path) -> dict:
    path = log_dir / RESULT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{RESULT_FILE} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_key(data: dict) -> str:
    missing = [k for k in ("target", "benchmark_version") if data.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"missing required benchmark fields: {', '.join(missing)}")
    return f"{data['target']}_v{data['benchmark_version']}"


def build_combo(data: dict) -> str:
    missing = [
        k for k in ("python_version", "rocm_version") if data.get(k) in (None, "")
    ]
    if missing:
        raise SystemExit(f"missing required combo fields: {', '.join(missing)}")
    gpu = (
        f"gpu_{data['gpu_count']}"
        if data.get("gpu_count") is not None
        else norm(data.get("runner", "unknown"))
    )
    return f"py{norm(data['python_version'])}-rocm{norm(data['rocm_version'])}-{gpu}"


def collect_benchmark_raw(data: dict):
    raw = {k: data[k] for k in BENCHMARK_RAW_FIELDS if data.get(k) is not None}
    return json.dumps(raw, sort_keys=True) if raw else None


def parse_maxtext_rocm_v1(data: dict):
    """Parse MaxText ROCm v1 benchmark metrics.
    Current public result.json provides:
      - run_code
      - cmp_code
      - distance_percent
      - expected_config_raw with baseline_ms and threshold_percent
    Parser generates standardized metrics:
      - status: 0=pass, 1=regression, 2=error
      - median_step_time_ms
      - distance_percent
    baseline and threshold_percent are stored on the run row.
    """
    run_code = int(data.get("run_code", 1))
    cmp_code = int(data.get("cmp_code", 1))
    status = 2 if run_code != 0 else 1 if cmp_code != 0 else 0
    rows = [("status", float(status))]
    expected = {}
    if data.get("expected_config_raw"):
        try:
            expected = json.loads(data["expected_config_raw"])
        except json.JSONDecodeError:
            expected = {}
    baseline = expected.get("baseline_ms")
    threshold = expected.get("threshold_percent")
    distance = data.get("distance_percent")
    if distance is not None:
        distance = float(distance)
        rows.append(("distance_percent", distance))
        if baseline not in (None, 0):
            baseline_value = float(baseline)
            # Current comparator is higher-is-better.
            # cmp_code already applies the threshold.
            observed = (
                baseline_value * (1.0 - distance / 100.0)
                if cmp_code != 0
                else baseline_value * (1.0 + distance / 100.0)
            )
            rows.append(("median_step_time_ms", observed))
    return (
        rows,
        float(baseline) if baseline is not None else None,
        float(threshold) if threshold is not None else None,
    )


METRIC_PARSERS = {
    "maxtext_rocm_v1": parse_maxtext_rocm_v1,
}


def parse_benchmark_metrics(data: dict, key: str):
    parser = METRIC_PARSERS.get(key)
    if parser is None:
        raise SystemExit(f"no metric parser registered for metric_key={key}")
    rows, baseline, threshold = parser(data)
    if not rows:
        raise SystemExit(f"metric parser returned no metrics for metric_key={key}")
    return rows, baseline, threshold


def build_run_row(
    data: dict,
    *,
    key: str,
    combo: str,
    baseline,
    threshold,
    run_tag: str,
    gpu_tag: str,
    artifact_uri: str | None,
):
    required = (
        "github_repository",
        "github_ref_name",
        "github_run_id",
        "is_nightly",
        "run_key",
        "target",
        "workload",
        "benchmark_version",
    )
    missing = [k for k in required if data.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"missing required run fields: {', '.join(missing)}")
    return {
        "artifact_uri": artifact_uri,
        "run_tag": run_tag,
        "gpu_tag": gpu_tag,
        "schema_version": int(data.get("schema_version") or 1),
        "github_repository": data["github_repository"],
        "github_ref_name": data["github_ref_name"],
        "github_ref": data.get("github_ref"),
        "github_event_name": data.get("github_event_name"),
        "github_run_url": data.get("github_run_url"),
        "github_sha": data.get("github_sha"),
        "github_run_id": int(data["github_run_id"]),
        "github_run_attempt": int(data.get("github_run_attempt") or 1),
        "github_run_number": data.get("github_run_number"),
        "github_workflow": data.get("github_workflow"),
        "github_job": data.get("github_job"),
        "is_nightly": data["is_nightly"],
        "run_key": data["run_key"],
        "combo": combo,
        "runner": data.get("runner"),
        "python_version": data.get("python_version"),
        "rocm_version": data.get("rocm_version"),
        "rocm_tag": data.get("rocm_tag"),
        "gpu_count": data.get("gpu_count"),
        "target": data["target"],
        "workload": data["workload"],
        "benchmark_version": int(data["benchmark_version"]),
        "metric_key": key,
        "baseline": baseline,
        "threshold_percent": threshold,
        "run_started_at": parse_iso_dt(data.get("run_started_at")),
        "run_completed_at": parse_iso_dt(data.get("run_completed_at")),
        "model_run_started_at": parse_iso_dt(data.get("model_run_started_at")),
        "model_run_completed_at": parse_iso_dt(data.get("model_run_completed_at")),
        "base_image_name": data.get("base_image_name"),
        "base_image_digest": data.get("base_image_digest"),
        "jax_packages_raw": data.get("jax_packages_raw"),
        "wheels_sha_raw": data.get("wheels_sha_raw"),
        "benchmark_raw_json": collect_benchmark_raw(data),
    }


def find_existing_run_id(cur, row: dict):
    cur.execute(
        """
       SELECT id
       FROM jax_ci_benchmark_runs
       WHERE github_repository = %s
         AND github_ref_name = %s
         AND is_nightly = %s
         AND run_key = %s
         AND metric_key = %s
         AND workload = %s
         AND combo = %s
       LIMIT 1
       """,
        (
            row["github_repository"],
            row["github_ref_name"],
            row["is_nightly"],
            row["run_key"],
            row["metric_key"],
            row["workload"],
            row["combo"],
        ),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def insert_benchmark_run(cur, row: dict) -> int:
    cur.execute(
        """
       INSERT INTO jax_ci_benchmark_runs (
         artifact_uri, run_tag, gpu_tag, schema_version,
         github_repository, github_ref_name, github_ref, github_event_name,
         github_run_url, github_sha, github_run_id, github_run_attempt,
         github_run_number, github_workflow, github_job,
         is_nightly, run_key, combo,
         runner, python_version, rocm_version, rocm_tag, gpu_count,
         target, workload, benchmark_version, metric_key,
         baseline, threshold_percent,
         run_started_at, run_completed_at,
         model_run_started_at, model_run_completed_at,
         base_image_name, base_image_digest,
         jax_packages_raw, wheels_sha_raw, benchmark_raw_json
       ) VALUES (
         %(artifact_uri)s, %(run_tag)s, %(gpu_tag)s, %(schema_version)s,
         %(github_repository)s, %(github_ref_name)s, %(github_ref)s, %(github_event_name)s,
         %(github_run_url)s, %(github_sha)s, %(github_run_id)s, %(github_run_attempt)s,
         %(github_run_number)s, %(github_workflow)s, %(github_job)s,
         %(is_nightly)s, %(run_key)s, %(combo)s,
         %(runner)s, %(python_version)s, %(rocm_version)s, %(rocm_tag)s, %(gpu_count)s,
         %(target)s, %(workload)s, %(benchmark_version)s, %(metric_key)s,
         %(baseline)s, %(threshold_percent)s,
         %(run_started_at)s, %(run_completed_at)s,
         %(model_run_started_at)s, %(model_run_completed_at)s,
         %(base_image_name)s, %(base_image_digest)s,
         %(jax_packages_raw)s, %(wheels_sha_raw)s, %(benchmark_raw_json)s
       )
       """,
        row,
    )
    return int(cur.lastrowid)


def sync_metrics(cur, names: list[str]) -> dict[str, int]:
    names = sorted(set(names))
    cur.executemany(
        """
       INSERT INTO jax_ci_benchmark_metrics (metric_name)
       VALUES (%s)
       ON DUPLICATE KEY UPDATE metric_name = VALUES(metric_name)
       """,
        [(name,) for name in names],
    )
    cur.execute(
        f"""
       SELECT id, metric_name
       FROM jax_ci_benchmark_metrics
       WHERE metric_name IN ({",".join(["%s"] * len(names))})
       """,
        names,
    )
    return {name: int(metric_id) for metric_id, name in cur.fetchall()}


def insert_metric_results(cur, run_id: int, metric_ids: dict[str, int], rows):
    cur.executemany(
        """
       INSERT INTO jax_ci_benchmark_results (run_id, metric_id, value)
       VALUES (%s, %s, %s)
       ON DUPLICATE KEY UPDATE value = VALUES(value)
       """,
        [(run_id, metric_ids[name], value) for name, value in rows],
    )


def ingest_benchmark_result(
    local_logs_dir: Path,
    *,
    run_tag: str,
    gpu_tag: str,
    artifact_uri: str | None,
):
    data = load_benchmark_result(local_logs_dir)
    key = metric_key(data)
    combo = build_combo(data)
    rows, baseline, threshold = parse_benchmark_metrics(data, key)
    run_row = build_run_row(
        data,
        key=key,
        combo=combo,
        baseline=baseline,
        threshold=threshold,
        run_tag=run_tag,
        gpu_tag=gpu_tag,
        artifact_uri=artifact_uri,
    )
    conn = connect()
    cur = conn.cursor()
    try:
        existing_run_id = find_existing_run_id(cur, run_row)
        if existing_run_id is not None:
            conn.rollback()
            print(
                "[DUPLICATE] benchmark result already exists: "
                f"run_id={existing_run_id} metric_key={key} "
                f"workload={run_row['workload']} combo={combo}"
            )
            return
        run_id = insert_benchmark_run(cur, run_row)
        metric_ids = sync_metrics(cur, [name for name, _ in rows])
        insert_metric_results(cur, run_id, metric_ids, rows)
        conn.commit()
        print(
            f"[summary] run_id={run_id} metric_key={key} "
            f"workload={run_row['workload']} combo={combo} metrics={len(rows)}"
        )
    except MySQLError as e:
        conn.rollback()
        if getattr(e, "errno", None) == 1062:
            print(
                f"[DUPLICATE] benchmark artifact already ingested: artifact_uri={artifact_uri}"
            )
            return
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Upload ROCm benchmark result.json to MySQL"
    )
    p.add_argument("--local_logs_dir", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--gpu-tag", required=True)
    p.add_argument("--artifact_uri", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_benchmark_result(
        Path(args.local_logs_dir),
        run_tag=args.run_tag,
        gpu_tag=args.gpu_tag,
        artifact_uri=args.artifact_uri,
    )
