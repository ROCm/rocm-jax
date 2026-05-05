"""Unit tests for ``regression.regression_classify``.

The new bucketing rule (matches the user spec):

  REGRESSION       failed today
                   AND failed in >= chronic_threshold of past window_days
                   nightlies
                   AND failed in at least one continuous-CI run within the
                   window or after today's nightly

  CHRONIC          chronic in nightly, but continuous CI shows it passing
  CHRONIC_PENDING  chronic in nightly, but no continuous-CI evidence at all
  NEWLY_BROKEN     today fail, prior nightly passed, not yet chronic
  KNOWN            today fail, prior nightly failed, not yet chronic
  NEW              today fail, no prior nightly run on file

Continuous-CI matching uses (nodeid, gpu_axis) only, ignoring the python
axis (the continuous workflow runs only py3.11, while nightly fans out
3.11-3.14).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_job  # noqa: E402
import regression  # noqa: E402


NIGHTLY = "CI - Wheel Tests (Nightly/Release)"
CONTINUOUS = "CI - Wheel Tests (Continuous)"
NIGHTLY_RE = r"Wheel Tests \(Nightly/Release\)"
CONTINUOUS_RE = r"Wheel Tests \(Continuous\)"

# The "today" anchor for the test fixtures.  Picking a fixed point makes
# the ``window_days`` arithmetic deterministic.
TODAY_DT = datetime(2026, 5, 8, 3, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stub(run_id: int, name: str, cell: str, failures, *,
          conclusion: str = "failure"):
    a = analyze_job.JobAnalysis(
        job_id=run_id * 10, name=name, matrix_cell=cell,
        conclusion=conclusion, duration_s=0,
    )
    a.failures = [
        analyze_job.Failure(nodeid=n, bucket="TEST_FAIL_FUNCTIONAL",
                            summary="...", excerpt="")
        for n in failures
    ]
    return a


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jax_triage_regression_"))
        self.db = self.tmp / "history.db"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def store(self, *, run_id: int, workflow_name: str, when: datetime,
              jobs, conclusion: str = "failure") -> None:
        regression.store_run(
            self.db, run_id=run_id, workflow_name=workflow_name,
            head_sha="cafebabe",
            run_date=when.date().isoformat(),
            created_at=_iso(when),
            conclusion=conclusion, html_url="",
            jobs=jobs,
        )

    def classify(self, today_run_id: int, **kwargs) -> dict:
        return regression.regression_classify(
            self.db, today_run_id=today_run_id,
            today_workflow_re=NIGHTLY_RE,
            continuous_workflow_re=CONTINUOUS_RE,
            **kwargs,
        )


class RegressionBucket(_Base):
    """REGRESSION = chronic-in-nightly + failing-in-continuous."""

    def test_chronic_plus_continuous_failing_is_REGRESSION(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_chronic_real"

        # Past 7 nights: this test fails on 5 of them (>= chronic_threshold=4).
        for i, run_id in enumerate([1001, 1002, 1003, 1004, 1005], start=1):
            day = TODAY_DT - timedelta(days=8 - i)  # days -7..-3
            failed = (i % 2 == 1)  # odd => fail (5 of 5? no, 3 of 5)
            self.store(run_id=run_id, workflow_name=NIGHTLY,
                       when=day,
                       jobs=[_stub(run_id, "nightly / 1gpu", cell,
                                   [nodeid] if failed else [])],
                       conclusion="failure" if failed else "success")
        # Force exactly 4 prior failures for clarity.
        for run_id, day_offset in [(1006, -2), (1007, -1)]:
            self.store(run_id=run_id, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(run_id, "nightly / 1gpu", cell, [nodeid])])
        # Today: also failing.
        today_run = 1100
        self.store(run_id=today_run, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(today_run, "nightly / 1gpu", cell, [nodeid])])
        # Continuous CI run within the window: also failing.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT - timedelta(hours=12),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [nodeid])])

        rr = self.classify(today_run)

        self.assertEqual(rr["regression"], [(nodeid, cell)])
        self.assertEqual(rr["chronic"], [])
        self.assertEqual(rr["chronic_pending"], [])
        self.assertEqual(rr["newly_broken"], [])
        self.assertGreaterEqual(
            len(rr["continuous_runs_used"]), 1,
            "continuous run within window should be used",
        )

    def test_chronic_with_continuous_passing_is_CHRONIC(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_chronic_env_specific"

        # 4 prior nightlies all failing (chronic).
        for i, day_offset in enumerate([-7, -5, -3, -1], start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu", cell,
                                   [nodeid])])
        # Today: failing.
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous: NOT failing this test (passed).
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT - timedelta(hours=6),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")

        rr = self.classify(1100)

        self.assertEqual(rr["chronic"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["chronic_pending"], [])

    def test_chronic_without_any_continuous_evidence_is_CHRONIC_PENDING(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_chronic_no_continuous_yet"

        for i, day_offset in enumerate([-7, -5, -3, -1], start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu", cell,
                                   [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Intentionally NO continuous runs in the DB.

        rr = self.classify(1100)

        self.assertEqual(rr["chronic_pending"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["chronic"], [])
        self.assertEqual(rr["continuous_runs_used"], [])

    def test_chronic_threshold_is_strict(self):
        """3-of-7 nights should NOT trigger chronic at default threshold=4."""
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_only_3_of_7"

        # 3 prior failures (below threshold of 4).
        for i, day_offset in enumerate([-6, -4, -2], start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu", cell,
                                   [nodeid])])
        # An intermediate passing run so prior_nightly is "passed yesterday".
        self.store(run_id=1099, workflow_name=NIGHTLY,
                   when=TODAY_DT - timedelta(days=1),
                   jobs=[_stub(1099, "nightly / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")
        # Today: failing.
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous CI: failing.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT - timedelta(hours=2),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["regression"], [],
                         "3 nights < default threshold of 4 should not "
                         "qualify as REGRESSION")
        # And lowering chronic_threshold to 3 promotes it.
        rr2 = self.classify(1100, chronic_threshold=3)
        self.assertEqual(rr2["regression"], [(nodeid, cell)])


class NewlyBrokenAndKnown(_Base):

    def test_passed_yesterday_failed_today_is_NEWLY_BROKEN(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_newly_broken"

        self.store(run_id=1000, workflow_name=NIGHTLY,
                   when=TODAY_DT - timedelta(days=1),
                   jobs=[_stub(1000, "nightly / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous evidence shouldn't matter for NEWLY_BROKEN.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT + timedelta(hours=2),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["newly_broken"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["known"], [])

    def test_failed_yesterday_failed_today_not_chronic_is_KNOWN(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_recently_broken"

        self.store(run_id=1000, workflow_name=NIGHTLY,
                   when=TODAY_DT - timedelta(days=1),
                   jobs=[_stub(1000, "nightly / 1gpu", cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # No continuous runs (irrelevant for KNOWN).

        rr = self.classify(1100)
        self.assertEqual(rr["known"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["newly_broken"], [])


class NewBucket(_Base):

    def test_no_prior_nightly_marks_NEW(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell,
                               ["tests/x.py::test_a", "tests/x.py::test_b"])])
        rr = self.classify(1100)
        self.assertEqual(rr["new"], [
            ("tests/x.py::test_a", cell),
            ("tests/x.py::test_b", cell),
        ])
        self.assertEqual(rr["regression"], [])
        self.assertIsNone(rr["prior_nightly_run_id"])


class GpuAxisCoarsening(_Base):
    """Continuous CI runs only py3.11 -- a py3.13 nightly failure should
    still match a py3.11 continuous failure on the same GPU axis."""

    def test_chronic_py313_nightly_matched_by_py311_continuous(self):
        nightly_cell = "1gpu-py3.13-rocm7.2.0"
        cont_cell    = "1gpu-py3.11-rocm7.2.0"
        nodeid       = "tests/x.py::test_py313_chronic"

        for i, day_offset in enumerate([-7, -5, -3, -1], start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu",
                                   nightly_cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", nightly_cell,
                               [nodeid])])
        # Continuous on py3.11 same GPU, also failing.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT - timedelta(hours=4),
                   jobs=[_stub(2001, "continuous / 1gpu", cont_cell,
                               [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["regression"], [(nodeid, nightly_cell)])
        self.assertEqual(rr["chronic"], [])


class WindowAndContinuousScope(_Base):

    def test_continuous_run_outside_window_is_ignored(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_old_continuous"

        for i, day_offset in enumerate([-7, -5, -3, -1], start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu", cell,
                                   [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous run from 30 days ago: outside the 7-day window.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT - timedelta(days=30),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(
            rr["continuous_runs_used"], [],
            "continuous run older than the window must not contribute")
        self.assertEqual(rr["chronic_pending"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])

    def test_no_failures_today_returns_empty(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")
        rr = self.classify(1100)
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["chronic"], [])
        self.assertEqual(rr["newly_broken"], [])
        self.assertEqual(rr["today_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
