"""Per-job analysis: fetch the log, parse failures, classify them.

Public entry point: :func:`analyze`. Everything else is internal but exposed
for unit testing / re-use.

The parser is deliberately regex-driven (no LLM) so a 12-job nightly takes
~5 seconds end-to-end. Failures that don't match any rule end up in the
``UNCATEGORIZED`` bucket -- those are the candidates a human (or a fan-out
subagent) should look at.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from github_client import GitHubClient


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

# Bucket vocabulary -- referenced by report.py and regression.py too.
BUCKETS = (
    "INFRA_RUNNER",
    "INFRA_TIMEOUT",
    "INFRA_OOM",
    "BUILD_FAIL",
    "IMPORT_FAIL",
    "TEST_FAIL_NUMERIC",
    "TEST_FAIL_HIP",
    "TEST_FAIL_FUNCTIONAL",
    "UNCATEGORIZED",
)


@dataclass
class Failure:
    nodeid: str                       # "tests/foo_test.py::test_bar[bf16]"
    bucket: str                       # one of BUCKETS
    summary: str = ""                 # short reason from short-summary
    excerpt: str = ""                 # last ~80 lines of traceback


@dataclass
class JobAnalysis:
    job_id: int
    name: str
    matrix_cell: str                  # canonical "1gpu-py3.11-rocm7.2.0"
    conclusion: str                   # success / failure / cancelled / skipped
    duration_s: int
    exit_step: Optional[int] = None
    failures: list[Failure] = field(default_factory=list)
    infra_events: list[str] = field(default_factory=list)
    log_path: Optional[str] = None    # filesystem path of the saved log
    pytest_totals: Optional[dict] = None  # {"failed": 9, "passed": 28261, ...}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failures"] = [asdict(f) for f in self.failures]
        return d


# ---------------------------------------------------------------------------
# Log fetch via `gh api`
# ---------------------------------------------------------------------------

def fetch_log(job_id: int, *, repo: str, dest_dir: Path,
              client: Optional[GitHubClient] = None) -> Path:
    """Download the raw log for one job and return the saved file path.

    Goes through :class:`GitHubClient` so it works equally well with a
    ``GITHUB_TOKEN`` env var (urllib path) or a logged-in ``gh`` CLI.
    Logs are sometimes gzipped; we transparently decompress those.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{job_id}.log"
    if out.exists() and out.stat().st_size > 0:
        return out

    client = client or GitHubClient()
    data = client.get_job_log(repo, job_id)
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    out.write_bytes(data)
    return out


# ---------------------------------------------------------------------------
# Pytest short-summary extraction
# ---------------------------------------------------------------------------

# pytest writes a banner "=== short test summary info ===" before the lines
# that look like "FAILED tests/x.py::y[param] - reason". The banner ends
# with another row of `=` characters at the next pytest section.
_SUMMARY_BANNER = re.compile(r"^=+\s*short test summary info\s*=+\s*$")
_SECTION_BANNER = re.compile(r"^=+ .+ =+\s*$")
_FAIL_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+)\s*(?:-\s*(.*))?$")

# GitHub Actions prefixes every line with a UTC timestamp.
_GHA_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")
# Pytest with --color=yes (the GH default) emits ANSI escapes around the
# section banners, the "FAILED"/"PASSED" keyword, and the test names. They
# look like "\x1b[31m" or the literal \x1b[31;1m form.  Strip them before
# trying to match anything.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Pytest's final summary line: "9 failed, 28261 passed, 28053 skipped in 1770.34s".
_TOTALS = re.compile(
    r"\b(?P<failed>\d+)\s+failed"
    r"(?:,\s*(?P<errors>\d+)\s+errors?)?"
    r"(?:,\s*(?P<passed>\d+)\s+passed)?"
    r"(?:,\s*(?P<skipped>\d+)\s+skipped)?",
)


def _clean(line: str) -> str:
    """Strip the GitHub Actions timestamp prefix and any ANSI escapes."""
    return _ANSI.sub("", _GHA_TS.sub("", line))


def extract_totals(log_text: str) -> Optional[dict]:
    """Parse the ``X failed, Y passed`` line that pytest prints at the end.
    Returns ``None`` if no such line is present (e.g. the runner died before
    pytest finished)."""
    last: Optional[dict] = None
    for line in log_text.splitlines():
        s = _clean(line)
        # Anchor on a line that contains both " failed" AND " in " <duration>,
        # to avoid matching "Failures (3):" style sub-headers.
        if " failed" not in s or " in " not in s:
            continue
        m = _TOTALS.search(s)
        if m:
            last = {k: int(v) for k, v in m.groupdict().items() if v}
    return last


def extract_short_summary(log_text: str) -> list[Failure]:
    """Return one :class:`Failure` per ``FAILED ...`` / ``ERROR ...`` line in
    the pytest short-summary block.  Buckets are filled in by :func:`classify`
    later -- this function only fills nodeid + raw summary.
    """
    failures: list[Failure] = []
    in_summary = False
    for line in log_text.splitlines():
        stripped = _clean(line)

        if _SUMMARY_BANNER.match(stripped):
            in_summary = True
            continue
        if in_summary and _SECTION_BANNER.match(stripped):
            in_summary = False
            continue
        if not in_summary:
            continue

        m = _FAIL_LINE.match(stripped)
        if m:
            _kind, nodeid, reason = m.group(1), m.group(2), (m.group(3) or "")
            failures.append(Failure(nodeid=nodeid, bucket="UNCATEGORIZED",
                                    summary=reason.strip()))
    return failures


# ---------------------------------------------------------------------------
# Traceback extraction
# ---------------------------------------------------------------------------

# Pytest's "FAILURES" section frames each failed test with
#   "_____ test_foo _____" / similar underscores.
_FAIL_HEADER = re.compile(r"^_+ (.+?) _+\s*$")


def extract_tracebacks(log_text: str) -> dict[str, str]:
    """Return a {short_test_name: excerpt} map from the FAILURES section.

    The matched key is the *short* nodeid that pytest prints in the underscore
    banner (e.g. ``test_bar``) which may not include the file path. The caller
    matches it against the full nodeid by suffix.
    """
    excerpts: dict[str, str] = {}
    cur_name: Optional[str] = None
    cur_buf: list[str] = []
    in_failures = False

    for line in log_text.splitlines():
        s = _clean(line)
        if re.match(r"^=+ FAILURES =+\s*$", s):
            in_failures = True
            continue
        if in_failures and _SECTION_BANNER.match(s) and "FAILURES" not in s:
            # End of FAILURES section. Flush.
            if cur_name is not None:
                excerpts[cur_name] = "\n".join(cur_buf[-80:])
            cur_name = None
            cur_buf = []
            in_failures = False
            continue
        if not in_failures:
            continue

        m = _FAIL_HEADER.match(s)
        if m:
            if cur_name is not None:
                excerpts[cur_name] = "\n".join(cur_buf[-80:])
            cur_name = m.group(1).strip()
            cur_buf = []
            continue
        if cur_name is not None:
            cur_buf.append(s)

    if cur_name is not None:
        excerpts[cur_name] = "\n".join(cur_buf[-80:])
    return excerpts


def attach_excerpts(failures: list[Failure], excerpts: dict[str, str]) -> None:
    """Best-effort match short-name -> full nodeid by suffix."""
    for f in failures:
        # nodeid like "tests/foo_test.py::test_bar[bf16]"
        # short name typically "test_bar[bf16]" or "test_bar"
        short = f.nodeid.split("::", 1)[-1]
        if short in excerpts:
            f.excerpt = excerpts[short]
            continue
        # Fallback: any short name ending with the same final token.
        token = short.split("[", 1)[0]
        for k, v in excerpts.items():
            if k.split("[", 1)[0] == token:
                f.excerpt = v
                break


# ---------------------------------------------------------------------------
# Classification (regex-first, plus exit-code heuristics)
# ---------------------------------------------------------------------------

# Order matters: more specific rules first. Each rule maps a bucket to a list
# of regex patterns matched against the failure summary OR excerpt OR full
# log (rule_scope decides which).
CLASSIFY_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    # (bucket, scope, pattern) -- scope in {"summary", "excerpt", "log"}
    ("INFRA_RUNNER",   "log",     re.compile(r"Executing the custom container implementation failed", re.I)),
    ("INFRA_RUNNER",   "log",     re.compile(r"ScriptExecutorError when trying to execute", re.I)),
    ("INFRA_TIMEOUT",  "log",     re.compile(r"The job running on runner .+ has exceeded the maximum execution time", re.I)),
    ("INFRA_OOM",      "log",     re.compile(r"\b(OOMKilled|hipErrorOutOfMemory|CUDA_ERROR_OUT_OF_MEMORY)\b")),
    ("INFRA_OOM",      "log",     re.compile(r"^Killed\b", re.M)),
    ("BUILD_FAIL",     "log",     re.compile(r"^ERROR: .+ failed to build", re.M | re.I)),
    ("BUILD_FAIL",     "log",     re.compile(r"bazel: ERROR", re.I)),
    ("IMPORT_FAIL",    "summary", re.compile(r"\bImportError\b|\bModuleNotFoundError\b")),
    ("IMPORT_FAIL",    "excerpt", re.compile(r"\bImportError\b|\bModuleNotFoundError\b")),
    ("TEST_FAIL_HIP",  "summary", re.compile(r"\bhipError|\bHIP error\b|\bROCm error\b|\bMIOPEN_STATUS|\brocBLAS_status", re.I)),
    ("TEST_FAIL_HIP",  "excerpt", re.compile(r"\bhipError|\bHIP error\b|\bROCm error\b|\bMIOPEN_STATUS|\brocBLAS_status", re.I)),
    ("TEST_FAIL_NUMERIC", "summary", re.compile(r"assert_allclose|Mismatched elements|tolerance|max abs diff", re.I)),
    ("TEST_FAIL_NUMERIC", "excerpt", re.compile(r"assert_allclose|Mismatched elements|tolerance|Max absolute difference", re.I)),
]


def classify_failure(f: Failure, *, log_text: str) -> str:
    for bucket, scope, pat in CLASSIFY_RULES:
        haystack = {"summary": f.summary, "excerpt": f.excerpt, "log": log_text}[scope]
        if haystack and pat.search(haystack):
            return bucket
    # Default: a failure that came from the pytest short-summary block but
    # didn't match any specific rule is a functional test failure.
    if f.summary or f.excerpt:
        return "TEST_FAIL_FUNCTIONAL"
    return "UNCATEGORIZED"


def classify_infra_only(log_text: str) -> list[str]:
    """Return any infra-level events, even when no test failure was parsed
    (e.g. the runner died before pytest started)."""
    out: list[str] = []
    for bucket, scope, pat in CLASSIFY_RULES:
        if not bucket.startswith("INFRA_") and bucket != "BUILD_FAIL":
            continue
        if scope != "log":
            continue
        if pat.search(log_text):
            out.append(bucket)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Matrix-cell parsing
# ---------------------------------------------------------------------------

# Job names look like:
#   "Pytest ROCm (JAX artifacts version = nightly) / 1gpu, ROCm 7.2.0, py3.11"
# We canonicalize the trailing matrix to "1gpu-py3.11-rocm7.2.0" so it sorts
# nicely and can be used as a stable key in the SQLite store.
_MATRIX_RE = re.compile(
    r"(?P<gpu>\d+gpu)\s*,\s*ROCm\s*(?P<rocm>[\d.]+)\s*,\s*py(?P<py>[\d.]+)",
    re.I,
)


def parse_matrix_cell(name: str) -> str:
    m = _MATRIX_RE.search(name)
    if not m:
        return name.replace(" ", "_")
    return f"{m['gpu']}-py{m['py']}-rocm{m['rocm']}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(job: dict, *, repo: str, log_dir: Path,
            client: Optional[GitHubClient] = None) -> JobAnalysis:
    """Analyze one job dict (as returned by ``gh api .../jobs``)."""
    job_id = int(job["id"])
    name = job["name"]
    cell = parse_matrix_cell(name)

    # Duration in seconds.
    try:
        from datetime import datetime
        s = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
        e = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
        duration_s = int((e - s).total_seconds())
    except Exception:
        duration_s = 0

    analysis = JobAnalysis(
        job_id=job_id, name=name, matrix_cell=cell,
        conclusion=job.get("conclusion") or "unknown",
        duration_s=duration_s,
    )

    if analysis.conclusion not in ("failure", "timed_out", "cancelled"):
        # Successful job -- nothing to analyze.
        return analysis

    log_path = fetch_log(job_id, repo=repo, dest_dir=log_dir, client=client)
    analysis.log_path = str(log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    # Failures from short-summary.
    failures = extract_short_summary(log_text)
    excerpts = extract_tracebacks(log_text)
    attach_excerpts(failures, excerpts)
    for f in failures:
        f.bucket = classify_failure(f, log_text=log_text)
    analysis.failures = failures

    # Infra events (independent of test failures).
    analysis.infra_events = classify_infra_only(log_text)

    # Sanity-check parsed failure count against pytest's own totals line.
    # If pytest reports N failures but we parsed != N, flag the gap so a
    # human knows the parser missed something (don't silently drop it).
    totals = extract_totals(log_text)
    analysis.pytest_totals = totals
    if totals and totals.get("failed", 0) and not failures:
        analysis.infra_events.append(
            f"PARSER_GAP: pytest reports {totals['failed']} failed but "
            f"the short-summary parser found 0")

    # exit_step: scrape the last "##[error]Process completed with exit code"
    # marker so we can flag jobs that died at step 7 (test execution) vs 8
    # (artifact upload) etc.
    m = re.findall(r"##\[error\]Process completed with exit code (\d+)", log_text)
    if m:
        # exit_step is approximated by the GitHub UI step index when present.
        # We can't reliably get the step number from the raw log, so we use
        # the count of step "##[group]" banners up to the error marker.
        analysis.exit_step = log_text.count("##[group]")

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Analyze one GH Actions job.")
    p.add_argument("job_id", type=int)
    p.add_argument("--repo", default=os.environ.get("TRIAGE_REPO", "jax-ml/jax"))
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--name", default="(unknown)",
                   help="Job name (only needed for matrix-cell parsing).")
    args = p.parse_args()
    client = GitHubClient()
    print(client.check_auth(), flush=True)
    job = {"id": args.job_id, "name": args.name, "conclusion": "failure",
           "started_at": "1970-01-01T00:00:00Z",
           "completed_at": "1970-01-01T00:00:00Z"}
    a = analyze(job, repo=args.repo, log_dir=Path(args.log_dir), client=client)
    print(json.dumps(a.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
