"""End-to-end smoke test on synthetic pytest logs (no GitHub call).

Feeds three synthetic logs through analyze_job + regression + report and
asserts the obvious invariants.
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

# Make the package importable when running this file from the tests dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_job
import regression
import report


# ---------------------------------------------------------------------------
# Synthetic log fixtures
# ---------------------------------------------------------------------------

# (1) A clean failing log: 2 numeric, 1 HIP failure.
LOG_NUMERIC_HIP = """\
2026-05-01T11:23:45.0000000Z ##[group]Run pytest
2026-05-01T11:23:45.0000000Z some warm-up output
2026-05-01T11:24:01.0000000Z ============================= test session starts ==============================
2026-05-01T11:24:01.0000000Z collected 100 items
2026-05-01T11:24:50.0000000Z =================================== FAILURES ===================================
2026-05-01T11:24:50.0000000Z _________________________ test_matmul_numeric[bf16] ___________________________
2026-05-01T11:24:50.0000000Z     def test_matmul_numeric(dtype):
2026-05-01T11:24:50.0000000Z >       np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)
2026-05-01T11:24:50.0000000Z E       AssertionError:
2026-05-01T11:24:50.0000000Z E       Not equal to tolerance rtol=0.01, atol=0.01
2026-05-01T11:24:50.0000000Z E       Mismatched elements: 8 / 16 (50%)
2026-05-01T11:24:50.0000000Z E       Max absolute difference: 1.2e-1
2026-05-01T11:24:50.0000000Z _________________________ test_softmax_numeric[fp16] _________________________
2026-05-01T11:24:50.0000000Z >       np.testing.assert_allclose(actual, expected)
2026-05-01T11:24:50.0000000Z E       Mismatched elements: 32 / 64 (50%)
2026-05-01T11:24:50.0000000Z _________________________ test_attention_hip ________________________________
2026-05-01T11:24:50.0000000Z >       out = jax.nn.dot_product_attention(q, k, v)
2026-05-01T11:24:50.0000000Z E       RuntimeError: HIP error: hipErrorInvalidValue
2026-05-01T11:24:50.0000000Z =========================== short test summary info ============================
2026-05-01T11:24:50.0000000Z FAILED tests/numeric_test.py::test_matmul_numeric[bf16] - AssertionError: tolerance
2026-05-01T11:24:50.0000000Z FAILED tests/numeric_test.py::test_softmax_numeric[fp16] - Mismatched elements
2026-05-01T11:24:50.0000000Z FAILED tests/attention_test.py::test_attention_hip - RuntimeError: HIP error
2026-05-01T11:24:51.0000000Z ##[error]Process completed with exit code 1
"""

# (2) An infra-runner failure with no test output.
LOG_INFRA_RUNNER = """\
2026-05-01T03:00:00.0000000Z Executing the custom container implementation failed.
2026-05-01T03:00:01.0000000Z ##[error]ScriptExecutorError when trying to execute: "Error: Job failed with exit code 1."
2026-05-01T03:00:01.0000000Z ##[error]Process completed with exit code 1
"""

# (3) An OOM during pytest.
LOG_OOM = """\
2026-05-01T05:00:00.0000000Z ##[group]Run pytest
2026-05-01T05:30:00.0000000Z =========================== short test summary info ============================
2026-05-01T05:30:00.0000000Z FAILED tests/big_test.py::test_huge - hipErrorOutOfMemory
2026-05-01T05:30:01.0000000Z Killed
2026-05-01T05:30:01.0000000Z ##[error]Process completed with exit code 137
"""


# ---------------------------------------------------------------------------
# Helpers: bypass `gh api` by writing logs to disk and short-circuiting fetch.
# ---------------------------------------------------------------------------

def _make_job(job_id, name, conclusion="failure", started=0, completed=60):
    return {
        "id": job_id,
        "name": name,
        "conclusion": conclusion,
        "started_at": "1970-01-01T00:00:00Z",
        "completed_at": f"1970-01-01T00:01:00Z",
    }


def _analyze_with_log(job_id, name, log_text, log_dir):
    """Mirror analyze_job.analyze without going to gh."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    log_path.write_text(log_text, encoding="utf-8")

    job = _make_job(job_id, name)
    a = analyze_job.JobAnalysis(
        job_id=job_id, name=name,
        matrix_cell=analyze_job.parse_matrix_cell(name),
        conclusion="failure", duration_s=60,
        log_path=str(log_path),
    )
    failures = analyze_job.extract_short_summary(log_text)
    excerpts = analyze_job.extract_tracebacks(log_text)
    analyze_job.attach_excerpts(failures, excerpts)
    for f in failures:
        f.bucket = analyze_job.classify_failure(f, log_text=log_text)
    a.failures = failures
    a.infra_events = analyze_job.classify_infra_only(log_text)
    return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class SmokeTest(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="jax_triage_smoke_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_short_summary_extraction(self):
        fs = analyze_job.extract_short_summary(LOG_NUMERIC_HIP)
        nodeids = [f.nodeid for f in fs]
        self.assertEqual(nodeids, [
            "tests/numeric_test.py::test_matmul_numeric[bf16]",
            "tests/numeric_test.py::test_softmax_numeric[fp16]",
            "tests/attention_test.py::test_attention_hip",
        ])

    def test_ansi_escapes_are_stripped(self):
        # Real GH Actions log lines: "<ts> \x1b[36m\x1b[1m=== short test
        # summary info ===\x1b[0m" and "\x1b[31mFAILED\x1b[0m
        # tests/foo.py::\x1b[1mtest_bar\x1b[0m - reason"
        ESC = "\x1b"
        log = (
            f"2026-05-01T11:00:00.0Z {ESC}[36m{ESC}[1m"
            f"=========================== short test summary info ===========================" 
            f"{ESC}[0m\n"
            f"2026-05-01T11:00:01.0Z {ESC}[31mFAILED{ESC}[0m "
            f"tests/x.py::{ESC}[1mFooTest::test_bar{ESC}[0m - "
            f"AssertionError: tolerance\n"
            f"2026-05-01T11:00:02.0Z {ESC}[31m========= {ESC}[31m{ESC}[1m"
            f"1 failed{ESC}[0m, {ESC}[32m100 passed{ESC}[0m"
            f"{ESC}[31m in 12.3s{ESC}[0m =========\n"
        )
        fs = analyze_job.extract_short_summary(log)
        self.assertEqual(len(fs), 1, msg=f"got: {[f.nodeid for f in fs]}")
        self.assertEqual(fs[0].nodeid, "tests/x.py::FooTest::test_bar")
        # Totals line must also parse despite ANSI noise.
        self.assertEqual(analyze_job.extract_totals(log)["failed"], 1)

    def test_traceback_attach(self):
        fs = analyze_job.extract_short_summary(LOG_NUMERIC_HIP)
        ex = analyze_job.extract_tracebacks(LOG_NUMERIC_HIP)
        analyze_job.attach_excerpts(fs, ex)
        self.assertTrue(any("hipErrorInvalidValue" in f.excerpt for f in fs),
                        msg=f"excerpts={ex}")

    def test_classification(self):
        a = _analyze_with_log(
            1001, "Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.11",
            LOG_NUMERIC_HIP, self.tmp / "logs")
        buckets = sorted(f.bucket for f in a.failures)
        self.assertEqual(
            buckets,
            ["TEST_FAIL_HIP", "TEST_FAIL_NUMERIC", "TEST_FAIL_NUMERIC"],
            msg=f"got: {[(f.nodeid, f.bucket) for f in a.failures]}")

    def test_infra_classification(self):
        a = _analyze_with_log(
            1002, "Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.12",
            LOG_INFRA_RUNNER, self.tmp / "logs")
        self.assertEqual(a.failures, [])
        self.assertIn("INFRA_RUNNER", a.infra_events)

    def test_oom_classification(self):
        a = _analyze_with_log(
            1003, "Pytest ROCm (...) / 4gpu, ROCm 7.2.0, py3.11",
            LOG_OOM, self.tmp / "logs")
        self.assertEqual([f.bucket for f in a.failures], ["INFRA_OOM"])
        self.assertIn("INFRA_OOM", a.infra_events)

    def test_matrix_cell_parse(self):
        for raw, want in [
            ("Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.11", "1gpu-py3.11-rocm7.2.0"),
            ("Pytest ROCm (...) / 8gpu, ROCm 7.2.0, py3.14", "8gpu-py3.14-rocm7.2.0"),
        ]:
            self.assertEqual(analyze_job.parse_matrix_cell(raw), want)

    def test_regression_diff_finds_new_and_chronic(self):
        db = self.tmp / "history.db"
        # Day 1..7 (chronic): a single failure repeats every night for 5 nights.
        for i in range(5):
            d = (date(2026, 4, 24) + timedelta(days=i)).isoformat()
            stub = analyze_job.JobAnalysis(
                job_id=2000 + i,
                name="Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.11",
                matrix_cell="1gpu-py3.11-rocm7.2.0",
                conclusion="failure", duration_s=10,
            )
            stub.failures = [analyze_job.Failure(
                nodeid="tests/x.py::test_chronic", bucket="TEST_FAIL_NUMERIC",
                summary="chronic numeric", excerpt="")]
            regression.store_run(
                db, run_id=2000 + i, workflow_name="WF", head_sha="abc",
                run_date=d, created_at=d + "T03:00:00Z",
                conclusion="failure", html_url="", jobs=[stub])

        # Today: chronic still fails AND a new test fails too.
        today = "2026-05-01"
        stub = analyze_job.JobAnalysis(
            job_id=3000, name="Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.11",
            matrix_cell="1gpu-py3.11-rocm7.2.0",
            conclusion="failure", duration_s=10,
        )
        stub.failures = [
            analyze_job.Failure("tests/x.py::test_chronic", "TEST_FAIL_NUMERIC", "...", ""),
            analyze_job.Failure("tests/y.py::test_new",     "TEST_FAIL_HIP",     "...", ""),
        ]
        regression.store_run(
            db, run_id=3000, workflow_name="WF", head_sha="def",
            run_date=today, created_at=today + "T03:00:00Z",
            conclusion="failure", html_url="", jobs=[stub])

        diff = regression.compute_diff(db, today=today, days=7,
                                       chronic_threshold=4)
        self.assertEqual(diff["new"],
                         [("tests/y.py::test_new", "1gpu-py3.11-rocm7.2.0")])
        self.assertEqual(diff["recurring"],
                         [("tests/x.py::test_chronic", "1gpu-py3.11-rocm7.2.0")])
        self.assertEqual(diff["recovered"], [])

    def test_full_report_renders(self):
        # Three jobs forming a tiny synthetic run.
        a1 = _analyze_with_log(
            1001, "Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.11",
            LOG_NUMERIC_HIP, self.tmp / "logs")
        a2 = _analyze_with_log(
            1002, "Pytest ROCm (...) / 1gpu, ROCm 7.2.0, py3.12",
            LOG_INFRA_RUNNER, self.tmp / "logs")
        a3 = _analyze_with_log(
            1003, "Pytest ROCm (...) / 4gpu, ROCm 7.2.0, py3.11",
            LOG_OOM, self.tmp / "logs")
        jobs = [a1, a2, a3]

        run_meta = {
            "run_id": 9999, "date": "2026-05-01",
            "head_sha": "deadbeef", "conclusion": "failure",
            "html_url": "https://example/9999",
            "created_at": "2026-05-01T03:00:00Z",
            "workflow_name": "Wheel Tests",
            "repo": "jax-ml/jax",
        }

        # Persist + diff against an empty history.
        db = self.tmp / "history.db"
        regression.store_run(
            db, run_id=9999, workflow_name="Wheel Tests",
            head_sha="deadbeef", run_date="2026-05-01",
            created_at="2026-05-01T03:00:00Z", conclusion="failure",
            html_url="", jobs=jobs)
        diff = regression.compute_diff(db, today="2026-05-01")

        out_dir = self.tmp / "report-out"
        paths = report.write_all(out_dir, run_meta=run_meta,
                                 jobs=jobs, diff=diff)

        self.assertTrue(paths["json"].exists())
        self.assertTrue(paths["markdown"].exists())
        self.assertTrue(paths["html"].exists())

        md = paths["markdown"].read_text(encoding="utf-8")
        # Buckets should appear.
        self.assertIn("TEST_FAIL_NUMERIC", md)
        self.assertIn("TEST_FAIL_HIP", md)
        self.assertIn("INFRA_RUNNER", md)
        self.assertIn("INFRA_OOM", md)
        # Heatmap header rows.
        self.assertIn("**1gpu**", md)
        self.assertIn("**4gpu**", md)
        # Per-job sections.
        self.assertIn("`1gpu-py3.11-rocm7.2.0`", md)
        # JSON parses.
        json.loads(paths["json"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
