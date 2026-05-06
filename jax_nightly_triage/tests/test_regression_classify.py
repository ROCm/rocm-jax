"""Unit tests for ``regression.regression_classify`` -- six-bucket model.

Buckets, in priority order:

    cancelled_infra | flaky | chronic | regression | known | newly_failed

Match keys: per-test buckets key on ``(nodeid, gpu, py)``.  ROCm is
ignored.  Stage-2 (continuous) only covers py3.11 nightly cells.
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

TODAY_DT = datetime(2026, 5, 8, 3, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stub(run_id: int, name: str, cell: str, failures, *,
          conclusion: str = "failure", flaky_tests: list[str] | None = None,
          infra_events: list[str] | None = None):
    a = analyze_job.JobAnalysis(
        job_id=run_id * 10 + hash(cell) % 1000,  # unique per cell
        name=name, matrix_cell=cell,
        conclusion=conclusion, duration_s=0,
    )
    a.failures = [
        analyze_job.Failure(nodeid=n, bucket="TEST_FAIL_FUNCTIONAL",
                            summary="...", excerpt="")
        for n in failures
    ]
    a.flaky_tests = list(flaky_tests or [])
    a.infra_events = list(infra_events or [])
    return a


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jax_triage_classify_"))
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


# ---------------------------------------------------------------------------
# regression bucket
# ---------------------------------------------------------------------------

class RegressionBucket(_Base):
    """`regression` = failed today + ALL prior nightlies in window passed."""

    def test_passed_in_all_prior_failed_today_is_regression(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_just_broke"

        # Seven prior nightlies, all passing this test (the cell ran but
        # the test wasn't in failures).
        for i, day_offset in enumerate(range(-7, 0), start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu", cell, [],
                                   conclusion="success")],
                       conclusion="success")
        # Today: same test fails.
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["regression"], [(nodeid, cell)])
        self.assertEqual(rr["known"], [])
        self.assertEqual(rr["newly_failed"], [])
        self.assertEqual(rr["flaky"], [])

    def test_continuous_also_failing_adds_stage2_badge(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_real_regression"

        for i, day_offset in enumerate(range(-7, 0), start=1):
            self.store(run_id=1000 + i, workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + i, "nightly / 1gpu", cell, [],
                                   conclusion="success")],
                       conclusion="success")
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous run AFTER today's nightly, also fails.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT + timedelta(hours=2),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["regression"], [(nodeid, cell)])
        # +continuous badge applied.
        self.assertEqual(rr["stage2_continuous_confirmed"],
                         [(nodeid, cell)])

    def test_cell_joined_mid_window_with_only_passes_is_newly_failed(self):
        """Cell didn't exist for the full window -- it ran in only some
        prior nightlies and never failed.  That's `newly-failed`, not
        `regression`."""
        cell = "1gpu-py3.11-rocm7.2.0"
        other_cell = "1gpu-py3.13-rocm7.2.0"  # different (gpu,py) -- ran instead
        nodeid = "tests/x.py::test_partial"

        # 7 prior nightly runs exist.  The cell-under-test only ran in
        # 3 of them; the other 4 nightlies ran a different cell so the
        # window itself is fully populated.
        ran_cell = {-1, -3, -5}
        for day_offset in range(-7, 0):
            cell_to_run = cell if day_offset in ran_cell else other_cell
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell_to_run, [],
                                   conclusion="success")],
                       conclusion="success")
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["newly_failed"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])


# ---------------------------------------------------------------------------
# known bucket
# ---------------------------------------------------------------------------

class KnownBucket(_Base):
    """`known` = failed today + failed in EVERY prior night the cell ran."""

    def test_failed_every_prior_night_is_known(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_old_friend"

        for day_offset in (-3, -2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["known"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["flaky"], [])

    def test_known_with_continuous_failing_adds_badge(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_old_friend"

        for day_offset in (-2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT + timedelta(hours=1),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["known"], [(nodeid, cell)])
        self.assertEqual(rr["stage2_continuous_confirmed"],
                         [(nodeid, cell)])


# ---------------------------------------------------------------------------
# chronic bucket (= "latest-continuous" renamed: failed today + continuous PASS)
# ---------------------------------------------------------------------------

class ChronicBucket(_Base):

    def test_failed_today_continuous_passes_is_chronic(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_only_in_nightly"

        # Three failing prior nights => without continuous evidence this
        # would be `known`.  Continuous covers (1gpu, py3.11) AND passes
        # the test, so it becomes `chronic`.
        for day_offset in (-3, -2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous after today: passes (job ran with conclusion success
        # and this nodeid is NOT in failures).
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT + timedelta(hours=4),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")

        rr = self.classify(1100)
        self.assertEqual(rr["chronic"], [(nodeid, cell)])
        self.assertEqual(rr["known"], [])
        self.assertEqual(rr["regression"], [])


# ---------------------------------------------------------------------------
# flaky bucket -- two paths: rerun-passed AND mixed-history
# ---------------------------------------------------------------------------

class FlakyBucket(_Base):

    def test_rerun_passed_is_flaky_even_with_failure(self):
        """If pytest-rerunfailures retried and passed, that's the
        canonical flaky signal -- highest priority."""
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_retry_passed"

        # The test is reported as failed in today's run AND in the
        # flaky_tests list.  flaky_tests should win.
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid],
                               flaky_tests=[nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["flaky"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["known"], [])
        self.assertEqual(rr["newly_failed"], [])

    def test_mixed_history_is_flaky(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_intermittent"

        # 2 prior fails, 2 prior passes (no full pattern).
        for day_offset, failed in [(-4, True), (-3, False),
                                   (-2, True),  (-1, False)]:
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell,
                                   [nodeid] if failed else [],
                                   conclusion="failure" if failed else "success")],
                       conclusion="failure" if failed else "success")
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["flaky"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["known"], [])


# ---------------------------------------------------------------------------
# newly-failed bucket -- partial or no history
# ---------------------------------------------------------------------------

class NewlyFailedBucket(_Base):

    def test_no_prior_data_at_all_is_newly_failed(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell,
                               ["tests/x.py::test_a",
                                "tests/x.py::test_b"])])
        rr = self.classify(1100)
        self.assertEqual(rr["newly_failed"], [
            ("tests/x.py::test_a", cell),
            ("tests/x.py::test_b", cell),
        ])
        self.assertEqual(rr["regression"], [])

    def test_partial_cell_history_all_passing_is_newly_failed(self):
        """Cell joined mid-window: ran in some prior nightlies (passing)
        but not all of them.  Failed today => newly-failed."""
        cell = "1gpu-py3.11-rocm7.2.0"
        other = "1gpu-py3.13-rocm7.2.0"
        nodeid = "tests/x.py::test_recent"

        # Window has 7 prior nightlies.  Cell ran in only 4 of them
        # (-4, -3, -2, -1) and passed; the remaining 3 nightlies ran a
        # different cell so the window itself is full.
        for day_offset in range(-7, 0):
            cell_to_run = cell if day_offset >= -4 else other
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell_to_run, [],
                                   conclusion="success")],
                       conclusion="success")
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        rr = self.classify(1100)
        self.assertEqual(rr["newly_failed"], [(nodeid, cell)])
        self.assertEqual(rr["regression"], [])


# ---------------------------------------------------------------------------
# cancelled_infra bucket -- per-cell, short-circuits per-test
# ---------------------------------------------------------------------------

class CancelledInfraBucket(_Base):

    def test_cancelled_job_lands_in_cancelled_infra(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [],
                               conclusion="cancelled")],
                   conclusion="failure")
        rr = self.classify(1100)
        self.assertEqual(len(rr["cancelled_infra"]), 1)
        self.assertEqual(rr["cancelled_infra"][0][0], cell)
        self.assertEqual(rr["cancelled_infra"][0][1], "cancelled")

    def test_infra_failure_with_no_pytest_signal(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [],
                               conclusion="failure",
                               infra_events=["INFRA_RUNNER"])],
                   conclusion="failure")
        rr = self.classify(1100)
        self.assertEqual(len(rr["cancelled_infra"]), 1)
        self.assertEqual(rr["cancelled_infra"][0][1], "infra_no_signal")
        self.assertIn("INFRA_RUNNER", rr["cancelled_infra"][0][2])

    def test_failure_with_parsed_failures_is_NOT_cancelled_infra(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_real_fail"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid],
                               infra_events=["INFRA_RUNNER"])])
        rr = self.classify(1100)
        self.assertEqual(rr["cancelled_infra"], [])
        # Falls through to newly-failed (no prior data).
        self.assertEqual(rr["newly_failed"], [(nodeid, cell)])


# ---------------------------------------------------------------------------
# Apple-to-apple matching
# ---------------------------------------------------------------------------

class AxisMatching(_Base):

    def test_py_axis_isolation_in_stage1(self):
        """A py3.11 failure must not be compared against py3.13 history."""
        cell11 = "1gpu-py3.11-rocm7.2.0"
        cell13 = "1gpu-py3.13-rocm7.2.0"
        nodeid = "tests/x.py::test_py_axis"

        # Lots of prior py3.13 history (both pass and fail).  This must
        # NOT influence the py3.11 classification.
        for day_offset in (-3, -2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell13, [nodeid])])
        # Today: only py3.11 fails.
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell11, [nodeid])])

        rr = self.classify(1100)
        # py3.11 has no prior data, so newly-failed (NOT known).
        self.assertEqual(rr["newly_failed"], [(nodeid, cell11)])
        self.assertEqual(rr["known"], [])

    def test_gpu_axis_isolation_in_stage1(self):
        """A 1gpu failure must not be compared against 4gpu history."""
        cell1 = "1gpu-py3.11-rocm7.2.0"
        cell4 = "4gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_gpu_axis"

        for day_offset in (-3, -2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 4gpu", cell4, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell1, [nodeid])])

        rr = self.classify(1100)
        self.assertEqual(rr["newly_failed"], [(nodeid, cell1)])
        self.assertEqual(rr["known"], [])

    def test_rocm_tag_change_does_NOT_wipe_history(self):
        """ROCm tag is dropped from the match key, so a tag bump should
        keep apple-to-apple history alignment for (gpu, py)."""
        old_cell = "1gpu-py3.11-rocm7.2.0"
        new_cell = "1gpu-py3.11-rocm7.3.0"
        nodeid = "tests/x.py::test_rocm_persists"

        # Prior nights ran on the old ROCm tag, all passing.
        for day_offset in range(-7, 0):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", old_cell, [],
                                   conclusion="success")],
                       conclusion="success")
        # Today bumps to a newer ROCm tag and fails.
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", new_cell, [nodeid])])

        rr = self.classify(1100)
        # History on old tag is reused for the new tag -> regression.
        self.assertEqual(rr["regression"], [(nodeid, new_cell)])

    def test_py313_nightly_NOT_matched_by_py311_continuous(self):
        """Continuous-CI runs only py3.11.  A py3.13 nightly failure
        must NOT match a py3.11 continuous run -- the py axis is part
        of the Stage-2 key."""
        nightly_cell = "1gpu-py3.13-rocm7.2.0"
        cont_cell    = "1gpu-py3.11-rocm7.2.0"
        nodeid       = "tests/x.py::test_py313"

        # Build a py3.13 known: failed every prior night the cell ran.
        for day_offset in (-2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu",
                                   nightly_cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu",
                               nightly_cell, [nodeid])])
        # Continuous on py3.11 also fails (would-be Stage-2 evidence).
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT + timedelta(hours=2),
                   jobs=[_stub(2001, "continuous / 1gpu",
                               cont_cell, [nodeid])])

        rr = self.classify(1100)
        # py3.13 stays in `known` but Stage-2 must NOT badge it: the
        # continuous evidence is on a different py axis.
        self.assertEqual(rr["known"], [(nodeid, nightly_cell)])
        self.assertEqual(rr["stage2_continuous_confirmed"], [])
        # And it must NOT become chronic either: continuous doesn't
        # cover (1gpu, py3.13).
        self.assertEqual(rr["chronic"], [])


# ---------------------------------------------------------------------------
# Stage-2 window: continuous runs strictly AFTER today only
# ---------------------------------------------------------------------------

class WindowAndContinuousScope(_Base):

    def test_continuous_run_BEFORE_today_is_ignored(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        nodeid = "tests/x.py::test_old_continuous"

        for day_offset in (-2, -1):
            self.store(run_id=1000 + day_offset + 10,
                       workflow_name=NIGHTLY,
                       when=TODAY_DT + timedelta(days=day_offset),
                       jobs=[_stub(1000 + day_offset + 10,
                                   "nightly / 1gpu", cell, [nodeid])])
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [nodeid])])
        # Continuous run from BEFORE today: must be ignored.
        self.store(run_id=2001, workflow_name=CONTINUOUS,
                   when=TODAY_DT - timedelta(hours=2),
                   jobs=[_stub(2001, "continuous / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")
        rr = self.classify(1100)
        self.assertEqual(rr["continuous_runs_used"], [],
                         "continuous run before latest nightly must be "
                         "ignored")
        # Without continuous evidence, falls through to known.
        self.assertEqual(rr["known"], [(nodeid, cell)])
        self.assertEqual(rr["chronic"], [])

    def test_no_failures_today_returns_empty(self):
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")
        rr = self.classify(1100)
        self.assertEqual(rr["regression"], [])
        self.assertEqual(rr["known"], [])
        self.assertEqual(rr["chronic"], [])
        self.assertEqual(rr["flaky"], [])
        self.assertEqual(rr["newly_failed"], [])
        self.assertEqual(rr["cancelled_infra"], [])
        self.assertEqual(rr["today_failure_count"], 0)

    def test_window_capped_at_seven(self):
        """Spec hard caps the window at 7 even if user passes more."""
        cell = "1gpu-py3.11-rocm7.2.0"
        self.store(run_id=1100, workflow_name=NIGHTLY,
                   when=TODAY_DT,
                   jobs=[_stub(1100, "nightly / 1gpu", cell, [],
                               conclusion="success")],
                   conclusion="success")
        rr = self.classify(1100, window_days=30)
        self.assertEqual(rr["window_days"], 7)


if __name__ == "__main__":
    unittest.main()
