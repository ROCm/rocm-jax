"""Upload Llama training summary results to MySQL database."""

import argparse
import ast
import json
import os
from datetime import date

# pylint: disable=import-error
import mysql.connector


def connect_to_db():
    """Connect to MySQL database."""
    return mysql.connector.connect(
        host=os.environ["ROCM_JAX_DB_HOSTNAME"],
        user=os.environ["ROCM_JAX_DB_USERNAME"],
        password=os.environ["ROCM_JAX_DB_PASSWORD"],
        database=os.environ["ROCM_JAX_DB_NAME"],
    )


# pylint: disable=too-many-statements, too-many-locals
def upload_llama_results(cli_args):
    """Load training summary results to MySQL."""
    rows = []
    year = date.today().year

    dataset = None
    base = None

    try:
        with open("training_summary.txt", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "train step" not in line.lower():
                    continue
                try:
                    # Parse timestamp
                    ts_code = line[1:5]  # MMDD
                    time_str = line[6:18]  # HH:MM:SS
                    mm, dd = int(ts_code[:2]), int(ts_code[2:])
                    timestamp = f"{year:04d}-{mm:02d}-{dd:02d} {time_str}"

                    # Parse step
                    step_idx = line.lower().index("train step")
                    colon_idx = line.index(":", step_idx)
                    step = int(line[step_idx + 10 : colon_idx].strip())

                    # Parse metrics
                    dict_str = line[colon_idx + 1 :].strip()
                    metrics = ast.literal_eval(dict_str)

                    if base is None:
                        first_key = list(metrics.keys())[0]
                        dataset = first_key.split("/", 1)[0]
                        base = f"{dataset}/ar_softmax_cross_entropy"

                    loss_text = metrics.get(f"{base}/text/loss")
                    loss_token = metrics.get(f"{base}/text/token_id/loss")
                    total_loss = metrics.get(f"{base}/total_loss")
                    acc = metrics.get(f"{base}/text/token_id/accuracy", {})
                    acc_top1 = acc.get("top_1", 0.0)
                    learning_rate = metrics.get("learning_rate", 0.0)

                    row = (
                        timestamp,
                        step,
                        loss_text,
                        loss_token,
                        total_loss,
                        acc_top1,
                        learning_rate,
                        json.dumps(metrics),  # Use JSON-safe string
                    )
                    rows.append(row)
                except (IndexError, ValueError) as e:
                    print(f"[Parse error] {e}")
    except FileNotFoundError:
        print("train_summary.txt not found.")
        return

    if not rows:
        print("No valid data found in train_summary.txt.")
        return

    cnx = None
    cursor = None

    try:
        cnx = connect_to_db()
        cursor = cnx.cursor()

        cursor.execute(
            """
            INSERT INTO perf_runs
            (github_run_id, tag, model_name, te_commit,
            jax_version, rocm_version, python_version,
            architecture, github_ref, trig_event, actor_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cli_args.github_run_id,
                cli_args.run_tag,
                cli_args.model_name,
                cli_args.te_commit,
                cli_args.jax_version,
                cli_args.rocm_version,
                cli_args.python_version,
                cli_args.runner_label,
                cli_args.github_ref,
                cli_args.trig_event,
                cli_args.actor_name,
            ),
        )

        run_id = cursor.lastrowid

        insert_sql = """
            INSERT INTO perf_metrics_step
            (run_id, ts, step, loss_text, loss_token,
            total_loss, accuracy_top_1, learning_rate, raw_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for row in rows:
            cursor.execute(insert_sql, (run_id,) + row)

        cnx.commit()
        print(f"Inserted {len(rows)} metrics for run_id={run_id}")

    except mysql.connector.Error as err:
        print(f"[Database error] {err}")
    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Upload LLAMA training summary metrics to MySQL"
    )

    p.add_argument("--run-tag", required=True, help="Run tag, e.g. ci-run")
    p.add_argument("--model-name", required=True, help="Model/workload, e.g. train_moe")
    p.add_argument("--te-commit", required=True, help="TE commit SHA, e.g. abc1234")
    p.add_argument("--jax-version", required=True, help="JAX version, e.g. 0.6.0")
    p.add_argument("--rocm-version", required=True, help="ROCm version, e.g. 7.2.0")
    p.add_argument("--python-version", required=True, help="Python version, e.g. 3.12")
    p.add_argument(
        "--github-run-id",
        required=True,
        type=int,
        help="Actions run id, e.g. 123456789",
    )
    p.add_argument("--github-ref", required=True, help="Git ref, e.g. master")
    p.add_argument("--trig-event", required=True, help="Trigger, e.g. schedule")
    p.add_argument("--actor-name", required=True, help="Actor, e.g. user_a")
    p.add_argument("--runner-label", required=True, help="Runner label, e.g. MI355")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_llama_results(args)
