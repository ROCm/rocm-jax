# `jax_nightly_triage` — generic GitHub Actions nightly triage with regression detection

Daily triage for any GitHub Actions workflow that runs `pytest`. Defaults
target [`jax-ml/jax`](https://github.com/jax-ml/jax) "CI - Wheel Tests
(Nightly/Release)" with the `Pytest ROCm` matrix, but every target is
overridable via flag or env var, so any team can drop the directory in,
point it at their nightly, and get a per-day Markdown / HTML / JSON report
that **separates real regressions from known-broken tests and from flakes**.

## What it does

1. **Discovers** the latest nightly run (or the specific `--run-id` you give it).
2. **Analyzes** every failing matrix cell in parallel (12 workers): downloads the
   raw GH Actions log, parses pytest's short-summary block + tracebacks,
   classifies failures into a small taxonomy (HIP / numeric / OOM / infra / …).
3. **Persists** every `(run, job, failure)` to a local SQLite store so future
   runs can diff against it.
4. **Cross-checks** today's failures against:
   - **Chronic history**: the past `window_days` nightlies (default 7).
   - **Continuous CI**: continuous runs within the same window or after today's nightly.
5. **Categorizes** every failure into one of:

   | Category | Meaning |
   |---|---|
   | 🚨 **REGRESSION** | failed today **and** failed in ≥ `chronic_threshold` of the past `window_days` nightlies (default 4-of-7) **and** failed in at least one continuous-CI run within the window or after today — **actionable**, multi-source confirmed. |
   | ♻️ **CHRONIC** | failed today + chronic in nightly, but continuous CI shows the test passing — likely env-specific or already fixed in HEAD. |
   | 🕒 **CHRONIC_PENDING** | failed today + chronic in nightly, but no continuous-CI evidence yet — re-run later. |
   | 🆕 **NEWLY_BROKEN** | failed today, **passed in the immediately prior nightly**, not yet chronic — something just changed in HEAD. |
   | ♻️ **KNOWN** | failed today **and** in the immediately prior nightly, but not yet chronic — recently broken. |
   | 🆕 **NEW** | failed today; no prior nightly in the database — first run, can't classify yet. |

6. **Renders** the verdict at the top of `report.md` / `report.html` and
   serializes the full machine-readable form to `summary.json`.

## Why it works

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

## Two-minute quick start

```bash
git clone <this repo>
cd jax_nightly_triage

# (1) Auth -- pick ONE:
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxx        # PAT
#  -- OR --
gh auth login                                            # if you use gh CLI

# (2) Sanity-check what auth is active and how much rate-limit you have:
python3 triage.py whoami
python3 triage.py rate-limit

# (3) Run today's nightly triage (auto-discovers the latest run):
python3 triage.py run

# (4) Open today's report:
xdg-open reports/$(date -u +%F)/report.html
```

## How to actually collect regressions (recommended workflow)

The categorizer needs **three** pieces of evidence to label a failure
correctly. Make sure they're all in the SQLite store.

### Step 1 — bootstrap a "yesterday" baseline (one-time)

The first triage run has no prior nightly to compare to, so everything
will be marked `NEW`. Replay yesterday's nightly first:

```bash
# Find yesterday's nightly run id:
python3 triage.py runs --limit 5
#       run_id  date        conclusion  url
# 25250121345   2026-05-02  cancelled   https://github.com/jax-ml/jax/actions/runs/...
# 25212524054   2026-05-01  failure     https://github.com/jax-ml/jax/actions/runs/...
# 25161628052   2026-04-30  failure     https://github.com/jax-ml/jax/actions/runs/...

# Ingest the previous nightly to populate the Stage 1 baseline:
python3 triage.py run --run-id 25161628052 --no-cross-check-continuous
```

`--no-cross-check-continuous` skips the continuous-CI fan-out for this
historical run; we only need its failure list as a baseline.

### Step 2 — triage today's nightly (the actual run)

```bash
python3 triage.py run --run-id 25212524054
```

What this does, in order (representative `stderr` output):

```
[nightly] 25212524054 (CI - Wheel Tests (Nightly/Release), 2026-05-01, failure)
  [1gpu-py3.11-rocm7.2.0] failure -- 6 failures, 0 infra events
  [1gpu-py3.12-rocm7.2.0] failure -- 6 failures, 0 infra events
  [4gpu-py3.13-rocm7.2.0] failure -- 9 failures, 0 infra events
  ... (12 jobs total, analyzed in parallel)
  14 continuous run(s) after nightly (0 already cached, 14 to ingest)
  [continuous] 25214205842 (CI - Wheel Tests (Continuous), 2026-05-01, success)
  [continuous] 25220188269 (CI - Wheel Tests (Continuous), 2026-05-01, failure)
  ...
  ingested 14 continuous run(s)
Wrote: reports/2026-05-01/summary.json
Wrote: reports/2026-05-01/report.md
Wrote: reports/2026-05-01/report.html
```

The Stage-2 bucket counts (REGRESSION / CHRONIC / CHRONIC_PENDING /
NEWLY_BROKEN / KNOWN / NEW) are not printed to stderr -- read them off
the top of `report.md` or `report.html`, or pull them from
`summary.json -> regression`.

### Step 3 — re-run later to resolve `CHRONIC_PENDING` verdicts

If no continuous-CI runs have completed in the cross-check window yet,
chronic-in-nightly failures land in `CHRONIC_PENDING` (the "we don't
know enough yet" bucket). Re-run later:

```bash
# A few hours later, after continuous CI has run another cycle:
python3 triage.py run --run-id 25212524054
# Already-cached continuous runs are skipped; only new ones are pulled.
# CHRONIC_PENDING entries get re-evaluated and become REGRESSION
# (continuous also failing) or CHRONIC (continuous passing).
```

### Step 4 — read the report

The Markdown report has the regression verdict **at the top**, before
all the other sections:

```markdown
## 🚨 Regression check (chronic ≥4-of-7 + continuous CI)

- Immediately prior nightly: `25161628052` (used for KNOWN vs NEWLY_BROKEN).
- Chronic window: 7 prior nights, threshold ≥4 of 7.
- Continuous-CI evidence: 14 run(s) within the window or after this nightly.

| Bucket | Definition | Count |
|---|---|---:|
| 🚨 **REGRESSION** | failed today + chronic in nightly (≥4-of-7) + failed in continuous CI | 4 |
| ♻️ **CHRONIC** | failed today + chronic in nightly, continuous CI shows it passing (likely env-specific or fixed in HEAD) | 7 |
| 🕒 **CHRONIC_PENDING** | failed today + chronic in nightly, but no continuous-CI evidence yet | 0 |
| 🆕 **NEWLY_BROKEN** | failed today, passed in the immediately prior nightly, not yet chronic | 25 |
| ♻️ **KNOWN** | failed today AND in the prior nightly, not yet chronic | 1 |
| 🆕 **NEW** | first time seen (no prior nightly to compare) | 0 |

### 🚨 REGRESSION (chronic in nightly + failing in continuous CI) (4)
| nodeid | cells affected |
|---|---|
| `tests/clear_backends_test.py::ClearBackendsTest::test_clear_backends` | 4gpu-py3.13-rocm7.2.0, 8gpu-py3.11-rocm7.2.0 |
| ...
```

`REGRESSION` is the only bucket meant to wake somebody up. `NEWLY_BROKEN`
is the second-most-interesting because it is a fresh delta against
yesterday's HEAD. `CHRONIC_PENDING` will be reclassified on the next run
once continuous-CI evidence lands.

## How the regression check works (the algorithm)

REGRESSION is the multi-source-confirmed bucket; the other buckets describe
weaker forms of evidence. The chronic-history check keys on the strict
`(nodeid, matrix_cell)` so "same test on the same matrix cell" is required.
The continuous-CI lookup uses `(nodeid, gpu_axis)` because continuous only
runs `py3.11` while the nightly fans out to `py3.{11,12,13,14}` — a nightly
failure on `1gpu-py3.13` is verifiable against `1gpu-py3.11` continuous runs.

```text
for each failing (nodeid, matrix_cell) in today's nightly:

  nights_failed   = # distinct prior nights in [today-window_days, today)
                    where (nodeid, matrix_cell) failed
  is_chronic      = nights_failed >= chronic_threshold  (default 4-of-7)
  in_continuous   = (nodeid, gpu_axis) failed in any continuous-CI run
                    within the window OR after today's nightly
  have_continuous = >=1 continuous-CI run is in the window or after today

  if is_chronic:
    if in_continuous                 -> REGRESSION         (actionable)
    elif have_continuous             -> CHRONIC            (HEAD passes)
    else                             -> CHRONIC_PENDING    (await continuous)
  elif prior_nightly is None         -> NEW                (first sighting)
  elif (nodeid, matrix_cell) in prior_nightly
                                     -> KNOWN              (failed yesterday too)
  else                               -> NEWLY_BROKEN       (passed yesterday)
```

The continuous workflow defaults to `r"Wheel Tests \(Continuous\)"`. Override
with `--continuous-workflow-re` or `TRIAGE_CONTINUOUS_RE` for any other shape
(e.g. a "presubmit" or "post-merge" workflow). Tune the chronic gate via
`--window-days N` and `--chronic-threshold K`. To get the most literal "fails
in any of the past N nights" reading, use `--chronic-threshold 1`.

## Subcommands

```text
python3 triage.py run            # default: triage today's nightly + write report
python3 triage.py workflows      # list workflows in --repo (find name regex)
python3 triage.py runs           # list recent runs of the resolved workflow
python3 triage.py whoami         # show which auth path is in use
python3 triage.py rate-limit     # show GitHub rate-limit headroom
```

`workflows` and `runs` are *discovery* helpers — use them once on a new
repo to find the right `--workflow-name-re` and `--continuous-workflow-re`,
then bake those into your `.env`.

```bash
# Find the workflows you want to target:
$ python3 triage.py workflows --repo jax-ml/jax | grep -i wheel
   138792004  active    CI - Wheel Tests (Nightly/Release)
   138792005  active    CI - Wheel Tests (Continuous)

# See what runs are available:
$ python3 triage.py runs --repo jax-ml/jax --limit 3
   25250121345  2026-05-02  cancelled   https://github.com/jax-ml/jax/actions/runs/25250121345
   25212524054  2026-05-01  failure     https://github.com/jax-ml/jax/actions/runs/25212524054
   25161628052  2026-04-30  failure     https://github.com/jax-ml/jax/actions/runs/25161628052
```

## CLI flags (full reference)

```text
$ python3 triage.py run --help
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

### N-night chronic diff (a separate, simpler signal)

| Flag | Env var | Default |
|---|---|---|
| `--window-days` | `TRIAGE_WINDOW_DAYS` | `7` |
| `--chronic-threshold` | `TRIAGE_CHRONIC_THRESHOLD` | `4` (of `window-days`) |

This produces the secondary `Recurring / Flaky / Recovered` table further
down the report — a rolling 7-night view, useful for spotting tests that
are *intermittently* broken vs cleanly regressing.

### Continuous-CI cross-check (Stage 2 regression detection)

| Flag | Env var | Default |
|---|---|---|
| `--cross-check-continuous` / `--no-cross-check-continuous` | — | enabled |
| `--continuous-workflow-re` | `TRIAGE_CONTINUOUS_RE` | `Wheel Tests \(Continuous\)` |
| `--continuous-job-prefix` | — | (mirrors `--job-prefix`) |
| `--continuous-max-runs` | — | `20` |

Disable with `--no-cross-check-continuous` when you want a quick triage
without the extra GitHub API budget (e.g. backfilling historical runs).

## Auth: tokens vs gh CLI

The `GitHubClient` in `github_client.py` resolves credentials in priority
order, and reports which mode is hot via `triage.py whoami`:

1. `GITHUB_TOKEN` env var
2. `GH_TOKEN` env var
3. `gh auth status` (subprocess fallback)

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

## Failure taxonomy

`analyze_job.CLASSIFY_RULES` matches against the failure summary line,
the traceback excerpt, or the full log:

| Bucket | Detection |
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

Add new buckets by extending `CLASSIFY_RULES` in `analyze_job.py`.

## Outputs

```
reports/
├── history.db                            # SQLite N-day history
└── 2026-05-01/                           # one dir per nightly
    ├── logs/<job_id>.log                 # raw GH Actions logs (cached)
    ├── summary.json                      # full machine-readable summary
    ├── report.md                         # markdown -- paste into issues
    └── report.html                       # heatmap dashboard
```

`summary.json` is the source of truth for downstream tooling. Top-level keys:

```json
{
  "run":  { "run_id": ..., "date": "2026-05-01", "head_sha": ..., "html_url": ... },
  "jobs": [ {"job_id": ..., "matrix_cell": ..., "failures": [...]}, ... ],

  "diff": {
    "today_count":       43,
    "window_days":       7,
    "chronic_threshold": 4,
    "new":               [["tests/x.py::test_y", "1gpu-py3.11-rocm7.2.0"], ...],
    "recurring":         [...],
    "flaky":             [...],
    "recovered":         [...],
    "prior_nights_seen": [
      {"nodeid": "tests/x.py::test_y",
       "matrix_cell": "1gpu-py3.11-rocm7.2.0",
       "nights_failed": 5}
    ]
  },

  "regression": {
    "regression":          [["tests/x.py::test_y", "1gpu-py3.11-rocm7.2.0"], ...],
    "chronic":             [...],
    "chronic_pending":     [...],
    "newly_broken":        [...],
    "known":               [...],
    "new":                 [...],
    "prior_nightly_run_id": 25161628052,
    "continuous_runs_used": [25214205842, 25220188269, ...],
    "today_failure_count":  43,
    "window_days":          7,
    "chronic_threshold":    4
  }
}
```

`diff` is the Stage-1 history-only partition; `regression` is the
Stage-2 multi-source classification. Each list under `regression` is a
sorted array of `[nodeid, matrix_cell]` pairs; a single
`(nodeid, matrix_cell)` appears in **exactly one** Stage-2 bucket.

To extract just the actionable regression list:

```bash
jq -r '.regression.regression[] | "\(.[0]) on \(.[1])"' \
   reports/$(date -u +%F)/summary.json
```

To inspect everything that's chronic but not yet confirmed in continuous CI:

```bash
jq -r '.regression.chronic_pending[] | "\(.[0]) on \(.[1])"' \
   reports/$(date -u +%F)/summary.json
```

## Cron / GH Actions

```cron
# Run every day at 07:15 UTC, after the JAX nightly typically finishes.
15 7 * * *   cd /opt/jax_nightly_triage && ./run.sh >> /var/log/triage.log 2>&1
# Re-run every 4 hours so CHRONIC_PENDING entries get reclassified
# (-> REGRESSION if continuous also fails, -> CHRONIC if continuous passes)
# as new continuous CI evidence arrives.
30 */4 * * * cd /opt/jax_nightly_triage && ./run.sh >> /var/log/triage.log 2>&1
```

`run.sh` reads `.env` if present, picks `GITHUB_TOKEN` over `gh` CLI, runs
`triage.py run`, and — if `TRACKING_ISSUE` is exported, e.g.
`my-fork/jax#42` — posts the markdown as a comment on that issue. The
comment-posting path uses pure `urllib` if `gh` isn't installed.

### Windows (Scheduled Tasks)

`run.ps1` is a one-to-one PowerShell port of `run.sh` (.env loading, auth
probe, pipeline, optional issue comment, 60-day cleanup). Requires
PowerShell 5.1+ (ships with Windows 10/11). Register it as a Scheduled
Task:

```text
Action:    powershell.exe
Arguments: -NoProfile -ExecutionPolicy Bypass -File C:\path\to\jax_nightly_triage\run.ps1
Trigger:   Daily at 07:15
```

Or invoke it directly:

```powershell
cd C:\path\to\jax_nightly_triage
$env:GITHUB_TOKEN = 'ghp_xxxxxxxx'   # or `gh auth login`
.\run.ps1
```

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
# 25 tests in ~1 second
```

Coverage:

- **log parsing** (timestamps, ANSI escapes, FAILED line variants, FAILURES
  section, totals line as a sanity check)
- **failure classification** (numeric / HIP / OOM / infra)
- **rolling-window diff** (Stage 1: NEW / RECURRING / FLAKY / RECOVERED
  across N nights)
- **multi-source regression classifier** (Stage 2: REGRESSION / CHRONIC /
  CHRONIC_PENDING / NEWLY_BROKEN / KNOWN / NEW — including the
  cross-Python-version match where a `py3.13` nightly failure is
  confirmed against `py3.11` continuous-CI runs on the same GPU axis)
- **HTTP client** (token mode, gh-CLI fallback, gzip decoding, retry-on-5xx,
  cross-origin redirect strips Authorization, helpful 404 message)
- **end-to-end report rendering** (markdown + JSON + HTML)

## Layout

```
jax_nightly_triage/
├── triage.py                       ← orchestrator (entry point, subcommands)
├── analyze_job.py                  ← log fetch + pytest parse + classification
├── regression.py                   ← SQLite history, N-night diff, 2-stage classifier
├── report.py                       ← markdown / JSON / HTML renderers
├── github_client.py                ← stdlib-only GitHub REST client (token / gh)
├── run.sh                          ← cron entrypoint (Linux / macOS)
├── run.ps1                         ← Scheduled Task entrypoint (Windows)
├── subagent_prompt.md              ← LLM fan-out template (optional)
├── requirements.txt                ← pure stdlib; no pip needed
├── .env.example                    ← env-var template
├── README.md
└── tests/
    ├── test_smoke.py               ← parser + report rendering
    ├── test_github_client.py       ← HTTP client (token / gh / redirects)
    └── test_regression_classify.py ← 2-stage regression classifier
```
