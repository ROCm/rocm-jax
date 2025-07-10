"""
upload_to_db.py

(Ruturaj4): This script reads a JSON file (`summary.json`) containing model
training performance metrics (step times, summary statistics) generated
during a CI job and uploads the data into a MySQL database table `ci_model_runs`.

It extracts metadata such as GitHub run ID, Python version, ROCm version, GPU architecture,
and JAX version (provided as command-line arguments), and stores per-model timing breakdowns
(`step0` to `step19`, plus min/median/mean statistics) along with the run metadata.

Environment variables required:
- ROCM_JAX_DB_HOSTNAME
- ROCM_JAX_DB_USERNAME
- ROCM_JAX_DB_PASSWORD
- ROCM_JAX_DB_NAME
- GITHUB_RUN_ID
- PYTHON_VERSION
- ROCM_VERSION
- GFX_VERSION
- JAX_VERSION
"""

import os
import json
import argparse
from typing import Any, Dict
from datetime import datetime

# pylint: disable=import-error
import mysql.connector
from mysql.connector import Error


def load_summary(filepath: str) -> Dict[str, Any]:
    """
    Loads the performance summary JSON from the given file path.

    Args:
        filepath: Path to the summary JSON file.

    Returns:
        Parsed JSON content as a dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def connect_to_database() -> mysql.connector.MySQLConnection:
    """
    Establishes and returns a MySQL database connection using environment variables.

    Returns:
        A MySQL connection object.

    Raises:
        RuntimeError: If environment variables are missing or connection fails.
    """
    try:
        host = os.environ["ROCM_JAX_DB_HOSTNAME"]
        user = os.environ["ROCM_JAX_DB_USERNAME"]
        password = os.environ["ROCM_JAX_DB_PASSWORD"]
        database = os.environ["ROCM_JAX_DB_NAME"]
    except KeyError as e:
        raise RuntimeError(f"Missing required environment variable: {e.args[0]}") from e

    try:
        return mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
        )
    except Error as e:
        raise RuntimeError(f"Failed to connect to MySQL database: {e}") from e


# pylint: disable=too-many-arguments,too-many-positional-arguments
def insert_model_run(
    cursor: mysql.connector.cursor.MySQLCursor,
    github_run_id: str,
    model_name: str,
    start_time: str,
    jax_version: str,
    rocm_version: str,
    gfx_version: str,
    python_version: str,
    model_data: Dict[str, Any],
) -> None:
    """
    Inserts a model run entry into the `ci_model_runs` table.

    Args:
        cursor: Active MySQL cursor.
        github_run_id: GitHub run identifier.
        model_name: Name of the model.
        start_time: Timestamp string.
        jax_version: JAX version.
        rocm_version: ROCm version.
        gfx_version: GPU architecture.
        python_version: Python version.
        model_data: Dictionary containing timing statistics and step data.
    """
    # (Ruturaj4) We only store the first 20 step times per model for two reasons:
    # 1. Empirically, the first 10–20 steps are sufficient to assess model performance.
    #    Median and variance typically stabilize within those early steps.
    # 2. Limiting to 20 steps keeps the DB schema fixed (and consistent over all models),
    #    and prevents excessive runtime during CI.
    steps = model_data.get("steps", [])
    step_times = [s["time"] for s in steps][:20]
    step_times += [None] * (20 - len(step_times))

    insert_query = f"""
        INSERT INTO ci_model_runs (
            github_run_id, model_name, start_time,
            jax_version, rocm_version, gfx_version, python_version,
            {', '.join([f'step{i}' for i in range(20)])},
            min_step_time, q25_step_time, median_step_time,
            mean_step_time, q75_step_time, max_step_time, steps_counted
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            {', '.join(['%s'] * 20)},
            %s, %s, %s, %s, %s, %s, %s
        )
    """

    values = [
        github_run_id,
        model_name,
        start_time,
        jax_version,
        rocm_version,
        gfx_version,
        python_version,
        *step_times,
        model_data["min_step_time"],
        model_data["q25_step_time"],
        model_data["median_step_time"],
        model_data["mean_step_time"],
        model_data["q75_step_time"],
        model_data["max_step_time"],
        model_data["steps_counted"],
    ]

    cursor.execute(insert_query, values)


def main():
    """
    Main entry point for uploading performance summary data to the database.
    """
    parser = argparse.ArgumentParser(
        description="Upload model timing summary to MySQL database."
    )
    parser.add_argument(
        "--summary-path", default="summary.json", help="Path to summary.json file"
    )
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--gfx-version", required=True)
    parser.add_argument("--jax-version", required=True)
    args = parser.parse_args()

    # Metadata from environment.
    summary = load_summary(args.summary_path)
    start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    conn = connect_to_database()
    cursor = conn.cursor()

    try:
        for model_name, model_data in summary.items():
            insert_model_run(
                cursor,
                github_run_id=args.github_run_id,
                model_name=model_name,
                start_time=start_time,
                jax_version=args.jax_version,
                rocm_version=args.rocm_version,
                gfx_version=args.gfx_version,
                python_version=args.python_version,
                model_data=model_data,
            )

        conn.commit()
        print("Data uploaded successfully to ci_model_runs table.")

    except Exception as e:
        print("Error occurred while uploading to DB:", e)
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
