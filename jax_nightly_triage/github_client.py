"""Minimal GitHub REST client with two interchangeable auth modes.

Mode 1 (preferred): set ``GITHUB_TOKEN`` (or ``GH_TOKEN``) to a PAT or a
fine-grained token. The client uses :mod:`urllib.request` directly -- no
external ``gh`` install needed.

Mode 2 (fallback): if no token is set but ``gh`` CLI is on ``PATH`` and
authenticated, the client shells out to ``gh api``.

If neither works, :meth:`GitHubClient.check_auth` raises with a clear
message that lists both options.

The reason both paths exist:
  - Mode 1 is what GitHub Actions, CI, cron jobs, and laptop users with a
    PAT need.  No homebrew install, no ``gh auth login`` flow.
  - Mode 2 is what developers who already use ``gh`` get for free.

There are NO third-party dependencies: stdlib ``urllib``, ``gzip``,
``json``, ``subprocess``, ``base64``.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import (
    HTTPRedirectHandler, HTTPSHandler, Request, build_opener,
)


# Default rate-limit-friendly retry policy.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_DEFAULT_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# Cross-origin auth-stripping redirect handler
# ---------------------------------------------------------------------------

class _AuthStrippingRedirectHandler(HTTPRedirectHandler):
    """Strip the ``Authorization`` header on cross-origin redirects.

    Python 3.13's stdlib started doing this automatically; on 3.10-3.12 the
    Bearer header is forwarded to the redirect target.  GitHub Actions logs
    302 to a signed ``*.githubusercontent.com`` / Azure blob URL that does
    NOT want the GitHub bearer token (it returns 401 if you send one,
    because the SAS token in the URL is the credential).  Without this
    handler, ``client.get_job_log()`` always 401s on Python <= 3.12.
    """

    def redirect_request(self, req: Request, fp, code: int, msg: str,
                         headers, newurl: str):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        # Compare host (and scheme) of original and new URLs; strip auth on
        # any cross-origin hop.
        try:
            old_host = urlsplit(req.full_url).netloc.lower()
            new_host = urlsplit(new_req.full_url).netloc.lower()
        except Exception:
            old_host = new_host = ""
        if old_host != new_host:
            for h in ("Authorization", "Cookie"):
                # Request headers are stored in two dicts: `headers` (lower-cased)
                # and `unredirected_hdrs` (preserved-case, set via add_unredirected_header).
                # Remove from both to be safe.
                new_req.headers.pop(h, None)
                new_req.headers.pop(h.lower(), None)
                new_req.unredirected_hdrs.pop(h, None)
                new_req.unredirected_hdrs.pop(h.lower(), None)
        return new_req


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GitHubAuthError(RuntimeError):
    """Raised when neither a token nor an authenticated gh CLI is available."""


class GitHubAPIError(RuntimeError):
    """Raised when an API call fails after retries."""


class GitHubClient:
    """Tiny GitHub REST client (no third-party deps)."""

    def __init__(self,
                 token: Optional[str] = None,
                 *,
                 base_url: str = "https://api.github.com",
                 user_agent: str = "jax-nightly-triage/1.0",
                 max_retries: int = 3,
                 retry_backoff_s: float = 2.0,
                 timeout_s: int = _DEFAULT_TIMEOUT_S):
        # Resolve token from kwarg or env in priority order. We strip
        # whitespace because newlines or trailing spaces in the env value
        # would otherwise raise a cryptic urllib "Invalid header value" with
        # the entire token leaking into the error message.
        raw = (
            token
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or ""
        )
        token_clean = raw.strip()
        # Sanity-check: GitHub tokens are single-line and short.  If the
        # value looks like a multiline blob (a likely paste of a help
        # message or stderr), drop it.
        if "\n" in token_clean or " " in token_clean or len(token_clean) > 256:
            token_clean = ""
        self.token = token_clean or None
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.timeout_s = timeout_s
        # On Python 3.13+ urllib auto-strips Authorization on cross-origin
        # redirects, but the JAX nightlies are routinely run on 3.10-3.12,
        # where the bearer header is forwarded.  GitHub Actions /logs 302s
        # to a signed Azure blob URL that returns 401 if it sees a bearer
        # token (the SAS token in the URL is the auth).  Our redirect
        # handler unconditionally strips Authorization on cross-origin hops.
        self._opener = build_opener(_AuthStrippingRedirectHandler(),
                                    HTTPSHandler())

    # ------------------------------------------------------------------
    # Auth resolution
    # ------------------------------------------------------------------

    def auth_mode(self) -> str:
        """Return ``"token"``, ``"gh-cli"``, or ``"none"`` -- a single source
        of truth for downstream code that wants to log which path is hot."""
        if self.token:
            return "token"
        if _gh_available() and _gh_authenticated():
            return "gh-cli"
        return "none"

    def check_auth(self) -> str:
        """Return a one-line summary of which path is used, or raise.

        Useful for orchestrator startup so the user sees what's about to happen
        before any API call burns rate-limit headroom.
        """
        mode = self.auth_mode()
        if mode == "token":
            tok_len = len(self.token) if self.token else 0
            tok_kind = _classify_token(self.token or "")
            return f"using GITHUB_TOKEN ({tok_kind}, len={tok_len})"
        if mode == "gh-cli":
            return "using gh CLI auth (no GITHUB_TOKEN env var found)"
        raise GitHubAuthError(
            "No GitHub credentials available.  Please choose one of:\n"
            "\n"
            "  (1) Personal access token (recommended, no extra install):\n"
            "      export GITHUB_TOKEN=ghp_xxx... # or ghs_/github_pat_...\n"
            "      Required scopes:\n"
            "        - public repos:  no scope needed for *public* read\n"
            "        - private repos: 'repo' (classic) OR fine-grained token\n"
            "                         with 'Actions: read' permission\n"
            "      Create one at: https://github.com/settings/tokens\n"
            "\n"
            "  (2) gh CLI (if you already use it):\n"
            "      gh auth login\n"
            "      Verify with: gh auth status\n"
        )

    # ------------------------------------------------------------------
    # Core HTTP
    # ------------------------------------------------------------------

    def get(self, path: str, *,
            accept: str = "application/vnd.github+json") -> bytes:
        """GET a REST endpoint and return raw response bytes (gunzipped if
        Content-Encoding was gzip).  Routes through the active auth mode."""
        mode = self.auth_mode()
        if mode == "none":
            self.check_auth()  # raises with the message above
        if mode == "token":
            return self._get_via_http(path, accept=accept)
        return self._get_via_gh(path, accept=accept)

    def get_json(self, path: str) -> Any:
        return json.loads(self.get(path))

    # ------------------------------------------------------------------
    # urllib path
    # ------------------------------------------------------------------

    def _get_via_http(self, path: str, *, accept: str) -> bytes:
        url = self._url(path)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            req = Request(url, headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": self.user_agent,
            })
            try:
                with self._opener.open(req, timeout=self.timeout_s) as resp:
                    body = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except HTTPError as e:
                if e.code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_exc = e
                    self._sleep_for(attempt, e)
                    continue
                # Surface helpful info for 401/403/404.
                detail = ""
                try:
                    detail = e.read().decode(errors="replace")[:300]
                except Exception:
                    pass
                raise GitHubAPIError(
                    f"GET {url} -> HTTP {e.code} {e.reason}: {detail}") from e
            except URLError as e:
                if attempt < self.max_retries:
                    last_exc = e
                    self._sleep_for(attempt, None)
                    continue
                raise GitHubAPIError(f"GET {url} failed: {e}") from e
        raise GitHubAPIError(f"GET {url} failed after retries: {last_exc}")

    def _sleep_for(self, attempt: int, http_err: Optional[HTTPError]) -> None:
        delay = self.retry_backoff_s * (2 ** attempt)
        if http_err is not None:
            # Honor Retry-After / X-RateLimit-Reset when present.
            ra = http_err.headers.get("Retry-After")
            if ra and ra.isdigit():
                delay = max(delay, int(ra))
            reset = http_err.headers.get("X-RateLimit-Reset")
            if reset and reset.isdigit():
                wait = int(reset) - int(time.time())
                if 0 < wait < 600:
                    delay = max(delay, wait)
        time.sleep(delay)

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    # ------------------------------------------------------------------
    # gh-cli path
    # ------------------------------------------------------------------

    def _get_via_gh(self, path: str, *, accept: str) -> bytes:
        cmd = ["gh", "api", "-H", f"Accept: {accept}", path]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace")[:400]
            raise GitHubAPIError(f"gh api {path} failed (rc={proc.returncode}): {err}")
        return proc.stdout

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def list_workflows(self, repo: str) -> list[dict]:
        return self.get_json(f"repos/{repo}/actions/workflows?per_page=100")["workflows"]

    def get_run(self, repo: str, run_id: int) -> dict:
        return self.get_json(f"repos/{repo}/actions/runs/{run_id}")

    def list_runs(self, repo: str, workflow_id: int, *,
                  branch: str = "main", per_page: int = 20,
                  event: Optional[str] = None) -> list[dict]:
        qs = f"per_page={per_page}&branch={quote(branch)}"
        if event:
            qs += f"&event={quote(event)}"
        return self.get_json(
            f"repos/{repo}/actions/workflows/{workflow_id}/runs?{qs}"
        )["workflow_runs"]

    def list_jobs(self, repo: str, run_id: int, *,
                  per_page: int = 100) -> list[dict]:
        return self.get_json(
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page={per_page}"
        )["jobs"]

    def get_job_log(self, repo: str, job_id: int) -> bytes:
        """Return the raw log bytes (already gunzipped if the server gzipped
        the response)."""
        return self.get(
            f"repos/{repo}/actions/jobs/{job_id}/logs",
            accept="application/vnd.github.raw",
        )

    def rate_limit(self) -> dict:
        """Return current rate-limit headroom -- handy for diagnostics."""
        return self.get_json("rate_limit")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gh_available() -> bool:
    try:
        proc = subprocess.run(["gh", "--version"], capture_output=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _gh_authenticated() -> bool:
    try:
        proc = subprocess.run(["gh", "auth", "status"],
                              capture_output=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _classify_token(tok: str) -> str:
    """Best-effort prefix-based classification of a token (for logging only).

    GitHub tokens have well-defined prefixes:
      - ghp_ : classic personal access token
      - ghs_ : server-to-server token (GH Apps, GH Actions GITHUB_TOKEN)
      - ghu_ : user-to-server token
      - github_pat_ : fine-grained personal access token
      - gho_ : OAuth user-to-server
    """
    for pfx, kind in (
        ("ghp_", "classic-PAT"),
        ("ghs_", "github-actions-token"),
        ("ghu_", "user-to-server"),
        ("github_pat_", "fine-grained-PAT"),
        ("gho_", "oauth"),
    ):
        if tok.startswith(pfx):
            return kind
    return "unknown-prefix"


# ---------------------------------------------------------------------------
# CLI entry: a tiny diagnostic command for first-time users.
#   python3 github_client.py whoami
#   python3 github_client.py rate-limit
#   python3 github_client.py workflows jax-ml/jax
# ---------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: github_client.py {whoami|rate-limit|workflows REPO}", flush=True)
        return 1
    cmd = argv[1]
    client = GitHubClient()
    if cmd == "whoami":
        print(client.check_auth())
        return 0
    if cmd == "rate-limit":
        rl = client.rate_limit()["resources"]["core"]
        print(json.dumps(rl, indent=2))
        return 0
    if cmd == "workflows":
        if len(argv) < 3:
            print("usage: github_client.py workflows REPO", flush=True)
            return 1
        repo = argv[2]
        for wf in client.list_workflows(repo):
            print(f"  {wf['id']:>12}  {wf['state']:<8}  {wf['name']}")
        return 0
    print(f"unknown subcommand: {cmd}", flush=True)
    return 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_cli(sys.argv))
