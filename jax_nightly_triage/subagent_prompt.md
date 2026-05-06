# Subagent prompt template — per-job analysis

Use this template when fanning out an LLM-driven re-analysis of a job whose
failures came back as `UNCATEGORIZED` from the regex parser, or when you
want a human-readable narrative for a single job.

The orchestrator (`triage.py`) does **not** invoke subagents by default —
the regex classifier is fast and deterministic. Reach for this prompt when
you want depth on one cell.

The subagent's job is **per-failure root-cause narrative** for a single
GitHub Actions job. The headline classification (the six buckets:
`cancelled_infra`, `flaky`, `chronic`, `regression`, `known`, `newly-failed`)
is owned by `regression.regression_classify` and is already written to
`reports/<DATE>/summary.json` before this prompt runs. The subagent
**reads** that classification and explains the *why* behind each failure;
it does **not** re-derive bucket assignments.

---

## Template

```
You are analyzing one failing GitHub Actions job in the JAX nightly suite.

Context:
  repo                = {{REPO}}                # default: jax-ml/jax
  run_id              = {{RUN_ID}}
  job_id              = {{JOB_ID}}
  job_name            = "{{JOB_NAME}}"
  matrix_cell         = "{{MATRIX_CELL}}"       # e.g. "1gpu-py3.11-rocm7.2.0"
  date                = {{DATE}}                # YYYY-MM-DD
  classification_json = {{CLASSIFICATION_PATH}} # reports/<DATE>/summary.json
                                                 # (the "classification" object inside it)

Tasks (do not commit; do not post to GitHub):
  1. Use `gh api -H 'Accept: application/vnd.github.raw' \
        repos/{{REPO}}/actions/jobs/{{JOB_ID}}/logs` to download the log
     into /tmp/{{JOB_ID}}.log. If gzipped, gunzip it.
  2. From the log:
     a. Extract the pytest "short test summary info" block.
     b. Extract the FAILURES section. For each failure, capture the last
        ~80 lines of its traceback.
     c. Note any infra-level events: container start failures, OOM,
        timeouts, ScriptExecutorError, runner errors, gh-action-runner
        cancellations.
     d. Note any pytest-rerunfailures markers (`RERUN`) -- these are the
        canonical flaky-by-retry signal.
  3. For each failure, set ``failure_category`` to ONE of:
       INFRA_RUNNER | INFRA_TIMEOUT | INFRA_OOM | BUILD_FAIL |
       IMPORT_FAIL | TEST_FAIL_FUNCTIONAL | TEST_FAIL_NUMERIC |
       TEST_FAIL_HIP | UNCATEGORIZED

     This is the **per-failure** category and is independent of the
     headline bucket.

  4. For each failure, look up its ``(nodeid, matrix_cell)`` in
     {{CLASSIFICATION_PATH}} and copy the ``headline_bucket`` field.
     Possible values (use the exact key as it appears in the JSON):

       cancelled_infra -- this matrix cell produced no pytest pass/fail
                          signal (cancelled / timed-out / infra-failed
                          before pytest); per-test classification was
                          skipped for the whole cell.
       flaky           -- pytest-rerunfailures retried-and-passed in this
                          job, OR the prior nightly window shows mixed
                          pass/fail for this (nodeid, gpu, py).
       chronic         -- failed today; same (gpu, py) covered by a
                          continuous-CI run after this nightly that
                          PASSED the test.  py3.11-only.
       regression      -- failed today; passed in EVERY prior nightly in
                          the window.
       known           -- failed today; failed in EVERY prior nightly the
                          cell ran in.
       newly_failed      -- failed today; cell or test had no full prior
                          history (cell joined mid-window, or test never
                          ran in earlier prior nightlies).

     If the headline_bucket is `regression` or `known` and the
     ``stage2_continuous_confirmed`` list in {{CLASSIFICATION_PATH}}
     contains the same (nodeid, matrix_cell) pair, also set
     ``continuous_confirmed: true`` on the failure.

  5. Produce a single JSON object on stdout with this shape:

     {
       "job_id":       <int>,
       "matrix_cell":  "<str>",
       "duration_min": <int>,
       "infra_events": ["INFRA_RUNNER", ...],
       "rerun_passed": ["tests/foo.py::test_a", ...],   # rerun-and-passed nodeids
       "failures": [
         {
           "nodeid": "tests/x.py::y[bf16]",
           "failure_category": "TEST_FAIL_NUMERIC",
           "headline_bucket": "regression" | "known" | "chronic" |
                              "flaky" | "newly_failed" | "cancelled_infra",
           "continuous_confirmed": true | false,
           "summary": "<one-line>",
           "root_cause_hypothesis": "<2-3 sentences with file/line refs>",
           "next_action": "bisect main between abc..def" |
                          "ping #rocm-infra"            |
                          "file flaky-test issue"       |
                          ...
         },
         ...
       ],
       "blast_radius": "<short note: only this cell? cross-cell?>",
       "confidence":   "high" | "medium" | "low"
     }

Constraints:
  - Stop after 4 unsuccessful gh API attempts and report the blocker.
  - Do not paste the full log; quote at most 20 lines per traceback.
  - If you cannot classify a failure, mark failure_category UNCATEGORIZED
    and explain in root_cause_hypothesis what's missing (truncated log,
    encrypted artifact, etc.).
  - DO NOT recompute the headline_bucket -- read it verbatim from
    {{CLASSIFICATION_PATH}}.  If the (nodeid, matrix_cell) pair is not
    present in any of the six bucket lists, set headline_bucket to null
    and say so in root_cause_hypothesis.
```

---

## How to fan out (orchestrator-side)

You can invoke this template by spawning one subagent per failing job, in
parallel, from the same chat turn. The subagents fan-in their per-job JSON
to a single message you compose:

```
For each failing job in `reports/<DATE>/summary.json`, launch one
generalPurpose subagent with the prompt template above (substitute the
{{...}} placeholders, including CLASSIFICATION_PATH=reports/<DATE>/summary.json).
Run them in parallel (run_in_background=true). After they all complete,
aggregate the per-job JSONs into reports/<DATE>/subagent_aggregate.json
and append a "Subagent narratives" section to report.md.
```

The classifier in `analyze_job.py` + `regression.py` is the cheap-and-fast
deterministic pass. Subagents are the expensive-but-deep narrative pass
for the failures the regex classifier can't categorize, or for any
headline bucket that needs a human-readable next-action. Keep the two
paths separate so a flaky network or LLM rate limit never breaks the
deterministic pipeline.
