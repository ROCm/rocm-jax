"""Unit tests for github_client.GitHubClient.

Exercises both auth modes against a local stdlib HTTP server, plus token-prefix
classification, retry-on-503, and gzip handling. No network calls.
"""
from __future__ import annotations

import gzip
import http.server
import json
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import github_client as gc


# ---------------------------------------------------------------------------
# Stub HTTP server
# ---------------------------------------------------------------------------

class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Tiny request handler programmable per-test via class-level routes."""

    routes: dict = {}     # path -> (status, body_bytes, headers, n_503_first)
    counters: dict = {}   # path -> request_count

    def log_message(self, *_args, **_kwargs):  # silence test output
        pass

    def do_GET(self):
        path = self.path
        _StubHandler.counters[path] = _StubHandler.counters.get(path, 0) + 1
        spec = _StubHandler.routes.get(path)
        if spec is None:
            self.send_response(404); self.end_headers(); return

        status, body, headers, n_503_first = spec
        if _StubHandler.counters[path] <= n_503_first:
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return

        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Server:
    def __init__(self):
        self.port = _free_port()
        self.httpd = http.server.HTTPServer(("127.0.0.1", self.port), _StubHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TokenClassification(unittest.TestCase):
    def test_known_prefixes(self):
        self.assertEqual(gc._classify_token("ghp_abc"), "classic-PAT")
        self.assertEqual(gc._classify_token("ghs_abc"), "github-actions-token")
        self.assertEqual(gc._classify_token("github_pat_abc"), "fine-grained-PAT")
        self.assertEqual(gc._classify_token("gho_abc"), "oauth")
        self.assertEqual(gc._classify_token("xxx"), "unknown-prefix")


class AuthResolution(unittest.TestCase):
    def test_token_env_picked_up(self):
        os.environ.pop("GH_TOKEN", None)
        os.environ["GITHUB_TOKEN"] = "ghp_test_token_value_xxx"
        try:
            client = gc.GitHubClient()
            self.assertEqual(client.auth_mode(), "token")
            self.assertIn("classic-PAT", client.check_auth())
        finally:
            os.environ.pop("GITHUB_TOKEN", None)

    def test_no_auth_raises_with_helpful_message(self):
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GH_TOKEN", None)
        client = gc.GitHubClient()
        # Force the gh-cli probes to return False so we test the "none" path.
        original_avail, original_authed = gc._gh_available, gc._gh_authenticated
        gc._gh_available = lambda: False
        gc._gh_authenticated = lambda: False
        try:
            with self.assertRaises(gc.GitHubAuthError) as ctx:
                client.check_auth()
            msg = str(ctx.exception)
            self.assertIn("GITHUB_TOKEN", msg)
            self.assertIn("gh auth login", msg)
        finally:
            gc._gh_available, gc._gh_authenticated = original_avail, original_authed


class HTTPMode(unittest.TestCase):
    """Run a stub HTTP server and exercise the urllib path end-to-end."""

    def setUp(self):
        _StubHandler.routes = {}
        _StubHandler.counters = {}
        os.environ["GITHUB_TOKEN"] = "ghp_test_token"

    def tearDown(self):
        os.environ.pop("GITHUB_TOKEN", None)

    def test_get_json_happy(self):
        with _Server() as srv:
            payload = {"workflows": [{"id": 1, "name": "X", "state": "active"}]}
            _StubHandler.routes["/repos/o/r/actions/workflows?per_page=100"] = (
                200, json.dumps(payload).encode(),
                {"Content-Type": "application/json"}, 0,
            )
            client = gc.GitHubClient(base_url=srv.base_url, retry_backoff_s=0)
            workflows = client.list_workflows("o/r")
            self.assertEqual(workflows[0]["name"], "X")

    def test_retry_on_503_then_success(self):
        with _Server() as srv:
            _StubHandler.routes["/repos/o/r/actions/runs/42"] = (
                200, json.dumps({"id": 42}).encode(),
                {"Content-Type": "application/json"}, 2,  # fail twice with 503
            )
            client = gc.GitHubClient(base_url=srv.base_url, retry_backoff_s=0,
                                     max_retries=3)
            run = client.get_run("o/r", 42)
            self.assertEqual(run["id"], 42)
            self.assertEqual(_StubHandler.counters["/repos/o/r/actions/runs/42"], 3)

    def test_gzip_response_is_decompressed(self):
        with _Server() as srv:
            raw = b"FAILED tests/x.py::test_y - boom\n" * 100
            gz = gzip.compress(raw)
            _StubHandler.routes["/repos/o/r/actions/jobs/9/logs"] = (
                200, gz, {"Content-Encoding": "gzip"}, 0,
            )
            client = gc.GitHubClient(base_url=srv.base_url, retry_backoff_s=0)
            data = client.get_job_log("o/r", 9)
            self.assertEqual(data, raw)

    def test_authorization_header_is_sent(self):
        captured = {}
        original = _StubHandler.do_GET
        def capture(self):
            captured["auth"] = self.headers.get("Authorization")
            captured["accept"] = self.headers.get("Accept")
            captured["api_ver"] = self.headers.get("X-GitHub-Api-Version")
            original(self)
        _StubHandler.do_GET = capture
        try:
            with _Server() as srv:
                _StubHandler.routes["/foo"] = (
                    200, b"{}", {"Content-Type": "application/json"}, 0)
                client = gc.GitHubClient(base_url=srv.base_url, retry_backoff_s=0)
                client.get("foo")
            self.assertEqual(captured["auth"], "Bearer ghp_test_token")
            self.assertEqual(captured["accept"], "application/vnd.github+json")
            self.assertEqual(captured["api_ver"], "2022-11-28")
        finally:
            _StubHandler.do_GET = original

    def test_cross_origin_redirect_strips_authorization(self):
        """Simulate the GitHub-Actions /logs flow: api.github.com 302's to a
        different host whose handler must NOT see the bearer token."""
        captured = {}
        original = _StubHandler.do_GET

        def _handler_with_capture(self):
            captured.setdefault(self.path, []).append(
                self.headers.get("Authorization"))
            original(self)

        _StubHandler.do_GET = _handler_with_capture
        try:
            with _Server() as redirect_target_srv, _Server() as api_srv:
                # On the redirect target, the request must NOT carry Bearer.
                _StubHandler.routes["/blob/abc"] = (
                    200, b"raw log payload",
                    {"Content-Type": "text/plain"}, 0,
                )
                # api.github.com -> 302 to the blob host (different netloc).
                redirect_url = f"{redirect_target_srv.base_url}/blob/abc"
                _StubHandler.routes["/repos/o/r/actions/jobs/9/logs"] = (
                    302, b"", {"Location": redirect_url}, 0,
                )
                client = gc.GitHubClient(base_url=api_srv.base_url,
                                         retry_backoff_s=0)
                data = client.get_job_log("o/r", 9)
            self.assertEqual(data, b"raw log payload")
            # The blob endpoint must have been called WITHOUT Authorization.
            self.assertEqual(captured.get("/blob/abc"), [None],
                             msg=f"captured={captured}")
            # The api endpoint must have been called WITH Authorization.
            self.assertEqual(
                captured.get("/repos/o/r/actions/jobs/9/logs"),
                ["Bearer ghp_test_token"])
        finally:
            _StubHandler.do_GET = original

    def test_404_surfaces_helpful_error(self):
        with _Server() as srv:
            _StubHandler.routes["/missing"] = (
                404, json.dumps({"message": "Not Found"}).encode(),
                {"Content-Type": "application/json"}, 0,
            )
            client = gc.GitHubClient(base_url=srv.base_url, retry_backoff_s=0,
                                     max_retries=0)
            with self.assertRaises(gc.GitHubAPIError) as ctx:
                client.get("missing")
            self.assertIn("404", str(ctx.exception))
            self.assertIn("Not Found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
