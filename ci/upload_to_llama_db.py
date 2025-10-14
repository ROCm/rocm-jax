"""Upload Llama training summary results to MySQL database."""

import os
import ast
import json
from datetime import date

# pylint: disable=import-error
import mysql.connector


def connect_to_db():
    return mysql.connector.connect(
        host=os.environ["ROCM_JAX_DB_HOSTNAME"],
        user=os.environ["ROCM_JAX_DB_USERNAME"],
        password=os.environ["ROCM_JAX_DB_PASSWORD"],
        database=os.environ["ROCM_JAX_DB_NAME"],
    )


def upload_llama_results():
    rows = []
    year = date.today().year

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

                    loss_text = metrics.get("mnist/ar_softmax_cross_entropy/text/loss")
                    loss_token = metrics.get(
                        "mnist/ar_softmax_cross_entropy/text/token_id/loss"
                    )
                    total_loss = metrics.get(
                        "mnist/ar_softmax_cross_entropy/total_loss"
                    )
                    acc_top1 = (
                        metrics.get(
                            "mnist/ar_softmax_cross_entropy/text/token_id/accuracy"
                        )
                        or {}
                    ).get("top_1", 0.0)
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
                int(os.environ["GITHUB_RUN"]),
                "ci-run",
                "llama",
                "9a2257b",
                "060",
                "072",
                "312",
                "MI355",
                os.environ["GITHUB_REF"],
                os.environ["TRIG_EVENT"],
                os.environ["ACTOR_NAME"],
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


if __name__ == "__main__":
    upload_llama_results()
