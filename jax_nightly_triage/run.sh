#!/usr/bin/env bash
# Cron entrypoint: discover the latest nightly, triage, write reports,
# optionally post the markdown to a tracking issue.
#
# Auth: prefers $GITHUB_TOKEN, falls back to `gh auth status`. See
# `.env.example` for all variables this script reads.
#
# Crontab example (07:15 UTC, after the JAX nightly typically finishes):
#   15 7 * * *  /home/you/repos/jax_nightly_triage/run.sh \
#               >> /tmp/jax_triage.log 2>&1

set -euo pipefail

cd "$(dirname "$0")"

# Source a local .env if present (kept out of git).
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

# ---- Auth probe -------------------------------------------------------------
# Token wins; otherwise rely on gh CLI.
if [ -n "${GITHUB_TOKEN:-}" ] || [ -n "${GH_TOKEN:-}" ]; then
  echo "[run.sh] using GITHUB_TOKEN env var"
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "[run.sh] using gh CLI auth"
else
  cat >&2 <<'EOF'
[run.sh] ERROR: no GitHub credentials available.

Pick one:
  (1) export GITHUB_TOKEN=ghp_xxx...   (https://github.com/settings/tokens)
      For public repos no scope is needed; for private repos use 'repo' or
      a fine-grained token with Actions: read.
  (2) gh auth login                    (https://cli.github.com/)
EOF
  exit 1
fi

# ---- Run the pipeline -------------------------------------------------------
python3 triage.py run "$@"

# ---- Optionally post today's report to a tracking issue ---------------------
TODAY="$(date -u +%Y-%m-%d)"
REPORT="${TRIAGE_REPORTS_DIR:-reports}/${TODAY}/report.md"

if [ -f "$REPORT" ] && [ -n "${TRACKING_ISSUE:-}" ]; then
  echo "[run.sh] posting report to ${TRACKING_ISSUE}..."
  REPO_PART="${TRACKING_ISSUE%#*}"
  ISSUE_NUM="${TRACKING_ISSUE##*#}"
  if command -v gh >/dev/null 2>&1; then
    gh issue comment "$ISSUE_NUM" --repo "$REPO_PART" --body-file "$REPORT"
  else
    # Direct REST POST via curl when gh isn't installed.
    if [ -z "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]; then
      echo "[run.sh] cannot post comment: no token and no gh CLI"
    else
      TOK="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
      python3 - "$REPO_PART" "$ISSUE_NUM" "$REPORT" "$TOK" <<'PY'
import json, sys, urllib.request
repo, issue, path, token = sys.argv[1:5]
body = open(path).read()
req = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/issues/{issue}/comments",
    data=json.dumps({"body": body}).encode(),
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "jax-nightly-triage/1.0",
    },
    method="POST",
)
with urllib.request.urlopen(req) as r:
    print(json.loads(r.read())["html_url"])
PY
    fi
  fi
fi

# ---- Hygiene: keep only the last 60 days of report dirs ---------------------
find "${TRIAGE_REPORTS_DIR:-reports}" -mindepth 1 -maxdepth 1 -type d \
     -mtime +60 -exec rm -rf {} \; 2>/dev/null || true
