# `jax_nightly_triage` — generic GitHub Actions nightly triage with two-stage regression detection

Daily triage for any GitHub Actions workflow that runs `pytest`. Defaults
target [`jax-ml/jax`](https://github.com/jax-ml/jax) "CI - Wheel Tests
(Nightly/Release)" with the `Pytest ROCm` matrix, but every target is
overridable via flag or env var, so any team can drop the directory in,
point it at their nightly, and get a per-day Markdown / HTML / JSON report
that **separates real regressions from known-broken tests, from flakes,
and from cells that didn't actually produce a pytest signal**.

---

## TL;DR

```bash
# 1. Auth (pick one)
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxx        # PAT, OR
gh auth login                                            # gh CLI

# 2. Triage today's nightly
python3 triage.py run

# 3. Read the report
xdg-open reports/$(date -u +%F)/report.html
```

The headline of every report is the **classification table** — six
buckets, in priority order. The first matching bucket wins:

| Bucket | What it means | Action |
|---|---|---|
| 🛑 `cancelled_infra` | Cell didn't produce a pytest pass/fail signal (cancelled, timed-out, infra-failed before pytest) | Fix infra, re-run |
| ⚠️ `flaky` | (a) pytest-rerunfailures retried-and-passed in this job, OR (b) mixed pass/fail history in the prior window | Track if persistent |
| ♻️ `chronic` | Failed in latest nightly **and** the same `(gpu, py)` was covered by a continuous-CI run after the nightly that **passed** the test | Likely env-specific to nightly runner; not a HEAD regression |
| 🚨 `regression` | Failed in latest nightly **and** passed in **every** prior nightly in the 7-day window | Wake somebody up |
| 🔁 `known` | Failed in latest nightly **and** failed in **every** prior nightly the cell ran in | Already broken; track |
| 🆕 `newly_failed` | Failed in latest nightly **and** the cell or test had no full prior history (cell joined mid-window, or test never ran in earlier prior nightlies) | Investigate; not enough history yet |

`regression` and `known` rows get a **`+continuous`** badge if the same
test also failed in a continuous-CI run after the nightly — strongest
signal that something is broken in HEAD on multiple pipelines.

---

## What it does, end to end

1. **Discovers** the latest nightly run (or the specific `--run-id` you give it).
2. **Analyzes** every matrix cell in parallel (12 workers): downloads the
   raw GH Actions log, parses pytest's short-summary block + tracebacks,
   classifies each individual failure into a small per-failure category
   (`INFRA_RUNNER` / `INFRA_OOM` / `BUILD_FAIL` / `TEST_FAIL_NUMERIC` /
   `TEST_FAIL_HIP` / `TEST_FAIL_FUNCTIONAL` / etc.), and extracts any
   `pytest-rerunfailures` retry-passed tests (the canonical flaky-by-rerun
   signal).
3. **Persists** every `(run, job, failure, flaky-by-rerun)` to a local
   SQLite store so future runs can compare against it.
4. **Cross-checks** the latest nightly against:
   - **Stage 1 — prior nightly window:** the previous 7 nightly runs
     (cap is hard-coded; missing nights are simply ignored).
   - **Stage 2 — continuous CI:** continuous-CI runs created **strictly
     after** the latest nightly's `created_at`. Continuous covers only
     `py3.11`, so only `py3.11` nightly cells participate in Stage 2.
5. **Classifies** every failing test into exactly one of the six buckets
   above using the priority tree
   `cancelled_infra → flaky → chronic → regression → known → newly_failed`.
6. **Renders** the verdict at the top of `report.md` / `report.html` and
   serializes the full machine-readable form to `summary.json`.

---

## Why it's safe to deploy

- **No third-party Python deps.** Pure stdlib: `urllib`, `sqlite3`, `re`,
  `json`. Run on any host with Python 3.10+.
- **Two auth modes.** `GITHUB_TOKEN` env var (preferred — works in CI / cron /
  fresh boxes) **or** an authenticated `gh` CLI as fallback.
- **Robust log parser.** Handles GitHub Actions timestamp prefixes, ANSI
  color escapes, gzip transfer-encoding, and the cross-origin redirect
  from `/jobs/<id>/logs` to the signed Azure blob.
- **Idempotent SQLite cache.** Re-running on the same `run_id` is a no-op
  for already-ingested rows; you can re-run a triage later in the day to
  pick up new continuous-CI evidence cheaply.
- **Deterministic classifier.** No LLM in the hot path. The optional
  subagent narrative path (`subagent_prompt.md`) reads the deterministic
  classifier's output rather than re-deriving buckets, so a flaky LLM
  call never changes the verdict.

---

## How the classifier works

### Match keys (apple-to-apple)

| Stage | Key | Why |
|---|---|---|
| Stage 1 (history) | `(nodeid, gpu, py)` | A `1gpu, py3.11` failure is only compared to `1gpu, py3.11` history. ROCm tag is **dropped** so a tag bump (e.g. `7.2.0 → 7.3.0`) does not wipe a week of history. |
| Stage 2 (continuous) | `(nodeid, gpu, py)` | Continuous CI runs only `py3.11`. So only the `py3.11` nightly cells (across all GPU axes) can ever match continuous-CI evidence. `py3.12 / py3.13 / py3.14` nightly cells are Stage-1-only by construction. |

### Stage 1 (per-test, against prior nightlies)

For every failing `(nodeid, gpu, py)` in the latest nightly, the
classifier examines the prior 7 nightlies (or fewer, if we have fewer on
file). It computes:

- `J` = number of prior nightlies in window where the **job** for
  `(gpu, py)` ran with a usable pytest signal (success / failure).
- `F` = number of those nights where this `nodeid` is in `failures`.

| Condition | Stage-1 bucket |
|---|---|
| `J == 0` | `newly_failed` (cell didn't exist in window) |
| `J < N_priors` AND `F == 0` | `newly_failed` (cell joined mid-window, no failures) |
| `J == N_priors` AND `F == 0` | `regression` (cell ran every prior night, never failed) |
| `J > 0` AND `F == J` | `known` (every night the cell ran, this test failed) |
| `0 < F < J` | `flaky` (mixed pass/fail across the window) |

> **Storage caveat.** The SQLite store records *failures*, not the full
> set of executed tests. So we infer "test ran and passed" from "the
> job for `(gpu, py)` ran AND this test isn't in `failures`". This
> conflates "test ran and passed" with "test didn't exist yet in that
> night" when the cell DID run. For most JAX tests the collected set
> is stable across a week, so the conflation is rare; the precise fix
> is to start tracking executed nodeids per job (a separate table).

### Stage 2 (per-test, against continuous CI)

For each failing test on a `py3.11` cell, the classifier consults
continuous-CI runs whose `created_at > latest_nightly.created_at`:

| Condition | Stage-2 effect |
|---|---|
| Same `(gpu, py)` covered by continuous AND continuous **passed** the test | Promote to `chronic` (overrides Stage-1 verdict for `regression` / `known` / `newly_failed`) |
| Same `(gpu, py)` covered AND continuous **also failed** the test | Stay in Stage-1 bucket; tag with `+continuous` badge |
| `(gpu, py)` not covered by continuous | Stage-1 verdict stands; no badge |

### Flaky detection has two independent paths

1. **Log-local (deterministic):** `pytest-rerunfailures` writes
   `<nodeid> RERUN` lines and a final verdict. If the last verdict for a
   nodeid is `PASSED` after at least one `RERUN`, the test is recorded
   in `JobAnalysis.flaky_tests` and lands in the `flaky` bucket
   regardless of any other signal. This is the highest-priority bucket
   in the tree.
2. **Statistical:** if the prior nightly window shows mixed pass/fail
   (`0 < F < J`), the test goes in `flaky` even without a rerun
   signal.

### Cancelled / infra cells (per-cell, not per-test)

A cell goes in `cancelled_infra` if **any** of:

- `conclusion in {cancelled, timed_out, skipped}`, OR
- `conclusion == failure` AND `infra_events != []` AND zero failures
  parsed from the log.

Per-test classification is **skipped** for these cells — there is no
pytest signal to interpret.

---

## Two-minute quick start

```bash
git clone <this repo>
cd jax_nightly_triage

# 1) Auth (pick ONE)
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxx        # PAT
gh auth login                                            # OR gh CLI

# 2) Sanity-check
python3 triage.py whoami            # which auth path is active
python3 triage.py rate-limit        # API budget headroom

# 3) Triage today's nightly (auto-discovers latest run)
python3 triage.py run

# 4) Read the report
xdg-open reports/$(date -u +%F)/report.html
```

---

## Recommended daily workflow

The classifier needs three pieces of evidence in the SQLite store to
return a complete verdict:

1. The latest nightly (today's).
2. The previous 7 nightlies (Stage-1 evidence).
3. Continuous-CI runs created strictly after the latest nightly
   (Stage-2 evidence).

### Step 1 — bootstrap a baseline (one-time)

The first triage run has no prior nightlies, so every test that fails
will land in `newly_failed`. Replay a few previous nightlies first:

```bash
# Find recent nightly runs
python3 triage.py runs --limit 7
#       run_id  date        conclusion  url
# 25250121345   2026-05-02  cancelled   https://github.com/jax-ml/jax/actions/runs/...
# 25212524054   2026-05-01  failure     https://github.com/jax-ml/jax/actions/runs/...
# 25161628052   2026-04-30  failure     https://github.com/jax-ml/jax/actions/runs/...
# ...

# Ingest them into history (skip Stage-2 to save API budget)
for run in 25212524054 25161628052 ... ; do
  python3 triage.py run --run-id $run --no-cross-check-continuous
done
```

`--no-cross-check-continuous` skips the continuous-CI fan-out for the
historical runs — we only need their failure lists as Stage-1 evidence.

### Step 2 — triage today's nightly

```bash
python3 triage.py run
# (no --run-id => discovers the latest nightly)
```

Representative `stderr`:

```
[nightly] 25212524054 (CI - Wheel Tests (Nightly/Release), 2026-05-08, failure)
  [1gpu-py3.11-rocm7.2.0] failure -- 6 failures, 0 infra events
  [1gpu-py3.12-rocm7.2.0] failure -- 6 failures, 0 infra events
  [4gpu-py3.13-rocm7.2.0] failure -- 9 failures, 0 infra events
  ... (12 jobs total, analyzed in parallel)
  3 continuous run(s) after nightly (0 already cached, 3 to ingest)
  [continuous] 25214205842 (CI - Wheel Tests (Continuous), 2026-05-08, success)
  [continuous] 25220188269 (CI - Wheel Tests (Continuous), 2026-05-08, failure)
  ingested 3 continuous run(s)
Wrote: reports/2026-05-08/summary.json
Wrote: reports/2026-05-08/report.md
Wrote: reports/2026-05-08/report.html
```

### Step 3 — re-run later to pick up more continuous-CI evidence

If no continuous-CI runs landed between nightly N's completion and your
first triage of nightly N, all `py3.11` cells fall back to Stage-1 only
(they can be `regression` / `known` / `newly_failed` / `flaky` but not
`chronic`). Re-running an hour later is cheap:

```bash
python3 triage.py run
# Already-cached continuous runs are skipped; only newly-completed ones
# are pulled. Existing classifications are recomputed against the
# updated continuous evidence.
```

### Step 4 — read the report

The Markdown report has the classification at the top:

```markdown
## 🚦 Classification

- Prior nightly window: **7** days (5 prior nightly run(s) on file).
- Continuous-CI evidence: **3** run(s) strictly after the latest nightly.

| Bucket | Definition | Count |
|---|---|---:|
| 🛑 cancelled / infra | latest job for the cell produced no pytest signal | 1 |
| ⚠️ flaky | rerun-passed in latest, OR mixed pass/fail prior history | 4 |
| ♻️ chronic | failed today + same `(gpu, py)` passed in continuous CI | 7 |
| 🚨 regression | failed today + passed in **all** prior nightlies | 2 |
| 🔁 known | failed today + failed in **all** prior nights the cell ran | 12 |
| 🆕 newly-failed | failed today + cell or test had no full prior history | 0 |

### 🚨 regression (passed in all prior nightlies) (2)
| nodeid | cells affected |
|---|---|
| `tests/x.py::test_y` `+continuous` | 1gpu-py3.11-rocm7.2.0 |
| `tests/x.py::test_z` | 4gpu-py3.13-rocm7.2.0 |
```

`+continuous` next to a `regression` or `known` row means the same test
also failed in a continuous-CI run after the nightly — multi-source
confirmed.

`regression` is the only bucket meant to wake somebody up. `known +
continuous` is the second-most-actionable. Everything else is
informational.

---

## Subcommands

```text
python3 triage.py run            # default: triage latest nightly + write report
python3 triage.py workflows      # list workflows in --repo (find name regex)
python3 triage.py runs           # list recent runs of the resolved workflow
python3 triage.py whoami         # show which auth path is in use
python3 triage.py rate-limit     # show GitHub rate-limit headroom
```

`workflows` and `runs` are *discovery* helpers — use them once on a new
repo to find the right `--workflow-name-re` and `--continuous-workflow-re`,
then bake those into your `.env`.

```bash
# Find the workflows you want to target
python3 triage.py workflows --repo jax-ml/jax | grep -i wheel
#  138792004  active    CI - Wheel Tests (Nightly/Release)
#  138792005  active    CI - Wheel Tests (Continuous)

# See what runs are available
python3 triage.py runs --repo jax-ml/jax --limit 3
#  25250121345  2026-05-02  cancelled   https://github.com/jax-ml/jax/actions/runs/25250121345
#  25212524054  2026-05-01  failure     https://github.com/jax-ml/jax/actions/runs/25212524054
#  25161628052  2026-04-30  failure     https://github.com/jax-ml/jax/actions/runs/25161628052
```

---

## CLI flags (full reference)

```bash
python3 triage.py run --help
```

### Targeting the workflow + jobs

| Flag | Env var | Default |
|---|---|---|
| `--repo` | `TRIAGE_REPO` | `jax-ml/jax` |
| `--workflow-name-re` | `TRIAGE_WORKFLOW_RE` | `Wheel Tests \(Nightly/Release\)` |
| `--job-prefix` | `TRIAGE_JOB_PREFIX` | `Pytest ROCm` |
| `--branch` | `TRIAGE_BRANCH` | `main` |
| `--run-id` | — | (auto-discover latest) |

### Persistence + reporting paths

| Flag | Env var | Default |
|---|---|---|
| `--db` | `TRIAGE_DB` | `reports/history.db` |
| `--reports-dir` | `TRIAGE_REPORTS_DIR` | `reports` |
| `--workers` | — | `12` (parallel per-job analyzers) |
| `--no-store` | — | (off; stores by default) |
| `--print-md` | — | (off; just paths logged) |

### Stage-1 history window

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--window-days` | `TRIAGE_WINDOW_DAYS` | `7` | Hard-capped at 7 by the spec; values > 7 are clamped. Missing nights inside the window are silently ignored ("work with the data we have"). |

### Stage-2 continuous-CI cross-check

| Flag | Env var | Default |
|---|---|---|
| `--cross-check-continuous` / `--no-cross-check-continuous` | — | enabled |
| `--continuous-workflow-re` | `TRIAGE_CONTINUOUS_RE` | `Wheel Tests \(Continuous\)` |
| `--continuous-job-prefix` | — | (mirrors `--job-prefix`) |
| `--continuous-max-runs` | — | `20` (newest continuous runs to inspect) |

`--no-cross-check-continuous` only disables *ingest*. The classifier
still runs Stage 2 against whatever continuous evidence is already in
the SQLite store; if there is none, all `py3.11` cells fall back to
Stage-1 bucketing.

---

## Auth

The `GitHubClient` resolves credentials in priority order:

1. `GITHUB_TOKEN` env var
2. `GH_TOKEN` env var
3. `gh auth status` (subprocess fallback)

`triage.py whoami` prints which is hot.

### Token scopes

| Visibility of the target repo | Required scope |
|---|---|
| **Public** (e.g. `jax-ml/jax`) | none — any logged-in token works for read |
| **Private** (classic PAT) | `repo` |
| **Private** (fine-grained PAT) | `Actions: read`, `Contents: read` on the target repos |

Create one at https://github.com/settings/tokens. The client logs the
token *prefix* (`ghp_`, `ghs_`, `gho_`, `github_pat_`, `ghu_`) so you can
verify the right kind is in use, but never logs the full token.

### Running inside GitHub Actions

GH Actions auto-injects `GITHUB_TOKEN` for every workflow:

```yaml
- name: Triage last night
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    TRIAGE_REPO:  ${{ github.repository }}
  run: python3 jax_nightly_triage/triage.py run
```

### Running on a fresh box without `gh`

```bash
sudo apt install -y python3                            # 3.10+
echo "export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxx" >> ~/.profile
source ~/.profile
python3 triage.py whoami                               # using GITHUB_TOKEN (...)
```

---

## Per-failure category (independent of the headline bucket)

`analyze_job.CLASSIFY_RULES` matches against the failure summary line,
the traceback excerpt, or the full log. This is the per-failure category
column shown in the per-job tables; it's **independent** of the
six-bucket headline classification.

| Category | Detection |
|---|---|
| `INFRA_RUNNER` | `Executing the custom container implementation failed`, `ScriptExecutorError` |
| `INFRA_TIMEOUT` | step exceeded job timeout |
| `INFRA_OOM` | `OOMKilled`, `hipErrorOutOfMemory`, `^Killed`, exit 137 |
| `BUILD_FAIL` | `bazel: ERROR`, "failed to build" |
| `IMPORT_FAIL` | `ImportError`, `ModuleNotFoundError` |
| `TEST_FAIL_NUMERIC` | `assert_allclose`, `Mismatched elements`, `tolerance` |
| `TEST_FAIL_HIP` | `hipError`, `MIOPEN_STATUS`, `rocBLAS_status`, `HIP error` |
| `TEST_FAIL_FUNCTIONAL` | any other `FAILED ...` line in pytest short-summary |
| `UNCATEGORIZED` | none of the above matched |

Add new categories by extending `CLASSIFY_RULES` in `analyze_job.py`.

---

## Outputs

```
reports/
├── history.db                            # SQLite history (runs / jobs / failures / flakies / infra_events)
└── 2026-05-08/                           # one dir per nightly date
    ├── logs/<job_id>.log                 # raw GH Actions logs (cached)
    ├── summary.json                      # full machine-readable summary
    ├── report.md                         # markdown -- paste into issues
    └── report.html                       # heatmap dashboard
```

`summary.json` is the source of truth for downstream tooling. Top-level
keys:

```jsonc
{
  "run":  { "run_id": ..., "date": "2026-05-08", "head_sha": ..., "html_url": ... },

  "jobs": [
    {
      "job_id": ...,
      "matrix_cell": "1gpu-py3.11-rocm7.2.0",
      "conclusion": "failure",
      "failures":     [ {"nodeid": ..., "bucket": "TEST_FAIL_NUMERIC", ...}, ... ],
      "flaky_tests":  [ "tests/foo.py::test_a", ... ],   // pytest-rerunfailures rerun-passed
      "infra_events": [ "INFRA_OOM", ... ]
    }
  ],

  "classification": {
    "regression":      [["tests/x.py::test_y", "1gpu-py3.11-rocm7.2.0"], ...],
    "known":           [...],
    "chronic":         [...],
    "flaky":           [...],
    "newly_failed":      [...],
    "cancelled_infra": [
      {"matrix_cell": "1gpu-py3.11-rocm7.2.0", "reason": "cancelled", "events": []}
    ],
    "stage2_continuous_confirmed": [
      ["tests/x.py::test_y", "1gpu-py3.11-rocm7.2.0"]
    ],
    "prior_nightly_run_ids":  [25161628052, ...],
    "continuous_runs_used":   [25214205842, 25220188269, ...],
    "today_failure_count":    43,
    "today_flaky_count":      2,
    "today_cell_count":       12,
    "window_days":             7
  }
}
```

A single `(nodeid, matrix_cell)` appears in **exactly one** of
`regression / known / chronic / flaky / newly_failed`. Cells in
`cancelled_infra` are recorded at the cell level — none of their tests
appear in the per-test buckets (no pytest signal to interpret).

`stage2_continuous_confirmed` is the subset of `regression + known`
whose failure was also observed in a continuous-CI run after the
nightly. The report renders these with a `+continuous` badge.

### Useful jq one-liners

```bash
# All actionable regressions
jq -r '.classification.regression[] | "\(.[0]) on \(.[1])"' \
   reports/$(date -u +%F)/summary.json

# Multi-source-confirmed (regression + known with +continuous badge)
jq -r '.classification.stage2_continuous_confirmed[] | "\(.[0]) on \(.[1])"' \
   reports/$(date -u +%F)/summary.json

# Cells that didn't even run (cancelled / infra)
jq -r '.classification.cancelled_infra[] | "\(.matrix_cell): \(.reason)"' \
   reports/$(date -u +%F)/summary.json

# Tests that pytest-rerunfailures rescued (flaky-by-rerun)
jq -r '.jobs[] | .matrix_cell as $c | .flaky_tests[]? | "\(.) on \($c)"' \
   reports/$(date -u +%F)/summary.json
```

---

## Cron / GH Actions

```cron
# Daily at 07:15 UTC, after the JAX nightly typically finishes
15 7 * * *   cd /opt/jax_nightly_triage && ./run.sh >> /var/log/triage.log 2>&1

# Re-run every 4 hours so cells fall into `chronic` when continuous-CI
# evidence catches up.
30 */4 * * * cd /opt/jax_nightly_triage && ./run.sh >> /var/log/triage.log 2>&1
```

`run.sh` reads `.env` if present, picks `GITHUB_TOKEN` over `gh` CLI,
runs `triage.py run`, and — if `TRACKING_ISSUE` is exported, e.g.
`my-fork/jax#42` — posts the markdown as a comment on that issue. The
comment-posting path uses pure `urllib` if `gh` isn't installed.

### Windows (Scheduled Tasks)

`run.ps1` is a one-to-one PowerShell port of `run.sh` (.env loading,
auth probe, pipeline, optional issue comment, 60-day cleanup). Requires
PowerShell 5.1+ (ships with Windows 10/11). Register it as a Scheduled
Task:

```text
Action:    powershell.exe
Arguments: -NoProfile -ExecutionPolicy Bypass -File C:\path\to\jax_nightly_triage\run.ps1
Trigger:   Daily at 07:15
```

Or invoke directly:

```powershell
cd C:\path\to\jax_nightly_triage
$env:GITHUB_TOKEN = 'ghp_xxxxxxxx'   # or `gh auth login`
.\run.ps1
```

---

## Optional: per-job LLM narrative (subagent path)

`subagent_prompt.md` is a template for fanning out one LLM subagent per
failing job to produce a human-readable root-cause narrative. The
subagent **reads** the deterministic classifier's output from
`summary.json` and explains the *why* behind each failure; it does
**not** re-derive the headline bucket. Keeping the deterministic and
narrative paths separate means a flaky LLM call never changes the
verdict.

---

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
# 39 tests in ~3 seconds
```

Coverage:

- **log parsing** (timestamps, ANSI escapes, FAILED line variants,
  FAILURES section, totals line as a sanity check)
- **failure classification** (numeric / HIP / OOM / infra)
- **pytest-rerunfailures detection** (`extract_rerun_passed` returns
  only nodeids whose last verdict is PASSED after at least one RERUN)
- **six-bucket classifier** (`regression`, `known`, `chronic`, `flaky`
  via both rerun and statistical paths, `newly_failed`, `cancelled_infra`)
- **apple-to-apple matching** (gpu and py axis isolation in Stage 1;
  `py3.13` nightly does NOT match `py3.11` continuous evidence)
- **ROCm-tag tolerance** (a tag bump does not wipe history)
- **Stage-2 window discipline** (continuous runs strictly *after* the
  latest nightly only; older continuous runs are ignored)
- **window cap** (values > 7 are clamped to 7)
- **HTTP client** (token mode, gh-CLI fallback, gzip decoding,
  retry-on-5xx, cross-origin redirect strips Authorization, helpful 404
  message)
- **end-to-end report rendering** (markdown + JSON + HTML)

---

## Layout

```
jax_nightly_triage/
├── triage.py                       ← orchestrator (entry point, subcommands)
├── analyze_job.py                  ← log fetch + pytest parse + per-failure categorization + rerun detection
├── regression.py                   ← SQLite history + six-bucket classifier
├── report.py                       ← markdown / JSON / HTML renderers
├── github_client.py                ← stdlib-only GitHub REST client (token / gh)
├── run.sh                          ← cron entrypoint (Linux / macOS)
├── run.ps1                         ← Scheduled Task entrypoint (Windows)
├── subagent_prompt.md              ← LLM fan-out template (optional, reads summary.json)
├── requirements.txt                ← pure stdlib; no pip needed
├── .env.example                    ← env-var template
├── README.md
└── tests/
    ├── test_smoke.py               ← parser + report rendering + rerun detection
    ├── test_github_client.py       ← HTTP client (token / gh / redirects)
    └── test_regression_classify.py ← six-bucket classifier
```
