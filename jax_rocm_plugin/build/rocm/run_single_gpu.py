#!/usr/bin/env python3
# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Single GPU test runner for JAX with ROCm.

This script runs JAX tests on individual GPUs in parallel, excluding
multi-GPU tests that require multiple devices.
"""

# pylint: disable=too-many-lines

import os
import shutil
import sys
import json
import argparse
import threading
import subprocess
import re
import html
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict

# Add the configuration directory to Python path
sys.path.insert(0, "jax_rocm_plugin/build/rocm")

GPU_LOCK = threading.Lock()
LAST_CODE = 0
BASE_DIR = "./logs"
ALL_CRASHED_TESTS = []  # Global list to track all crashed tests


def _external_abort_plugin_dir() -> str:
    """Return absolute path to external /pytest-abort directory.

    We compute it relative to this repo root so it works regardless of cwd.
    Repo root is three levels up from this file:
      jax_rocm_plugin/build/rocm/run_single_gpu.py -> rocm-jax repo root
    """
    rocm_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(rocm_dir, "..", "..", ".."))
    return os.path.abspath(os.path.join(repo_root, "..", "pytest-abort"))


# Make external plugin importable in this process (for helpers).
_EXT_ABORT_PLUGIN_DIR = _external_abort_plugin_dir()  # pylint: disable=invalid-name
if os.path.isdir(_EXT_ABORT_PLUGIN_DIR):
    sys.path.insert(0, _EXT_ABORT_PLUGIN_DIR)

# Shared abort/report + sanitization logic lives in external plugin package.
from pytest_abort.abort_handling import (  # type: ignore  # pylint: disable=wrong-import-position
    handle_abort,
    sanitize_for_json,
)
from pytest_abort.crash_file import (  # type: ignore  # pylint: disable=wrong-import-position
    check_for_crash_file,
)
from pytest_abort.logs import (  # type: ignore  # pylint: disable=wrong-import-position
    ensure_logs_dir,
)
from pytest_abort.report_utils import (  # type: ignore  # pylint: disable=wrong-import-position
    generate_final_report as generate_final_report_plugin,
)


def detect_amd_gpus():
    """Detect number of AMD/ATI GPUs using rocminfo."""
    try:
        cmd = [
            "bash",
            "-c",
            ('rocminfo | egrep -c "Device Type:\\s+GPU"'),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        print("Warning: Could not detect GPUs using rocminfo, defaulting to 8")
        return 8


def clear_crash_file(crash_file_path):
    """Clear crash detection file if it exists."""
    if os.path.exists(crash_file_path):
        try:
            os.remove(crash_file_path)
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[WARNING] Could not remove crash file: {e}")
            return False
    return True


# pylint: disable=too-many-arguments,too-many-positional-arguments
def build_pytest_command(
    json_log_file,
    html_log_file,
    test_nodeids,
    continue_on_fail=False,
    reruns=3,
    deselect_tests=None,
):
    """Build pytest command for running tests.

    Args:
        json_log_file: Path to JSON report file
        html_log_file: Path to HTML report file
        test_nodeids: List of test node IDs to run
        continue_on_fail: Whether to continue on failure (don't use -x flag)
        reruns: Number of times to rerun failed tests
        deselect_tests: List of tests to deselect (for crash recovery)

    Returns:
        List of command arguments
    """
    if deselect_tests is None:
        deselect_tests = []

    cmd = [
        "python3",
        "-m",
        "pytest",
        "--json-report",
        f"--json-report-file={json_log_file}",
        f"--html={html_log_file}",
        "--reruns",
        str(reruns),
        "-v",
    ]

    # Add -x flag before test nodeids if needed
    if not continue_on_fail:
        cmd.append("-x")

    # Add all test nodeids
    cmd.extend(test_nodeids)

    # Add deselect flags for crashed tests
    for test in deselect_tests:
        cmd.extend(["--deselect", test])

    return cmd


def extract_test_name(path: str) -> str:
    """Return the base filename (without .py and without test suffixes)."""
    return os.path.splitext(os.path.basename(path.split("::")[0]))[0]


def _tests_dict_from_only_test_files(only_test_files: str):
    """Build a minimal tests_dict from a comma-separated list of test files.

    Each entry is run as a single pytest invocation (per GPU token), which is
    compatible with the existing crash retry logic (it will re-run the same file
    with `--deselect <crashed-nodeid>` when a hard crash is detected).
    """
    tests_dict = {}
    for raw in (only_test_files or "").split(","):
        p = raw.strip()
        if not p:
            continue
        if p.startswith("./"):
            p = p[2:]
        # Allow shorthand "tests/foo_test.py" -> "jax/tests/foo_test.py"
        if p.startswith("tests/"):
            p = f"jax/{p}"
        # Allow shorthand "foo_test.py" -> "jax/tests/foo_test.py"
        if "/" not in p and p.endswith(".py"):
            p = f"jax/tests/{p}"
        log_name = os.path.splitext(os.path.basename(p))[0]
        tests_dict[log_name] = [p]
    return tests_dict


def run_shell_command(cmd, shell=False, env_vars=None, timeout=3600):
    """Run a shell command and return the result.

    Args:
        cmd: Command to run
        shell: Whether to run in shell mode
        env_vars: Environment variables to set
        timeout: Timeout in seconds (default: 3600 = 1 hour)
    """
    if env_vars is None:
        env_vars = {}
    env = os.environ.copy()
    env.update(env_vars)

    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, env=env, check=False, timeout=timeout
        )
        if result.returncode != 0:
            print(f"FAILED - {' '.join(cmd)}")
            print(result.stderr.decode())

        return result.returncode, result.stderr.decode(), result.stdout.decode()
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT - Command exceeded {timeout}s: {' '.join(cmd)}")
        return 124, f"Timeout after {timeout}s", ""


def parse_test_log(log_file):
    """Parse the test module log file to extract test modules and functions."""
    test_files = set()
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            report = json.loads(line)
            if "nodeid" in report:
                module = report["nodeid"].split("::")[0]
                if module and ".py" in module:
                    prefix = "" if module.startswith("tests/") else "tests/"
                    test_files.add(os.path.abspath("./jax/" + prefix + module))
    return test_files


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def collect_testmodules(ignore_skipfile=True):
    """Collect all test modules, excluding multi-GPU tests."""

    # copy to jax as it's node ids for pytest.
    shutil.copy("ci/pytest_drop_test_list.ini", "jax/")

    # jax/pytest_drop_test_list has ingore list for multi-gpu
    # append skip list as -deselect if it exists
    if not ignore_skipfile and os.path.exists("ci/pytest_skips.ini"):
        with open(
            "./jax/pytest_drop_test_list.ini", "a", encoding="utf-8"
        ) as outfile, open("ci/pytest_skips.ini", encoding="utf-8") as infile:
            outfile.write(infile.read())

    pytest_cmd = [
        "python3",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--import-mode=prepend",
        "-c",
        "./jax/pytest_drop_test_list.ini",
        "jax/tests",
    ]

    # run pytest collection and save in file collected_tests.txt
    with open("collected_tests.txt", "w", encoding="utf-8") as f:
        subprocess.run(
            pytest_cmd, stdout=f, stderr=subprocess.PIPE, check=True, text=True
        )

    # create key value store for test_name, test-ids
    tests_count = 0
    tests_dict = defaultdict(list)
    with open("collected_tests.txt", "r", encoding="utf-8") as f:
        for test_id in f:
            if test_id.startswith(("tests/", "./tests/")):
                test_name = extract_test_name(test_id.strip())
                tests_dict[test_name].append(f"jax/{test_id.strip()}")
                tests_count += 1

    print(f"test-files={len(tests_dict)}, total number of tests={tests_count}")
    return tests_dict


def run_test(log_name, tests_list, gpu_tokens, continue_on_fail):
    """Run tests from a test module on an available GPU with crash recovery.

    Runs all tests from the file in one pytest call. If a crash is detected,
    re-runs the remaining tests with the crashed test deselected.
    """
    # pylint: disable=global-statement,global-variable-not-assigned
    global LAST_CODE
    global ALL_CRASHED_TESTS

    # Acquire GPU token
    with GPU_LOCK:
        if LAST_CODE != 0:
            return
        target_gpu = gpu_tokens.pop()

    try:
        env_vars = {
            "HIP_VISIBLE_DEVICES": str(target_gpu),
            "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
        }

        json_log_file = f"{BASE_DIR}/{log_name}_log.json"
        html_log_file = f"{BASE_DIR}/{log_name}_log.html"
        last_running_file = f"{BASE_DIR}/{log_name}_last_running.json"
        # Enable abort-detector plugin (writes this file per-test).
        env_vars["PYTEST_ABORT_LAST_RUNNING_FILE"] = os.path.abspath(last_running_file)
        # Ensure external plugin is importable in the pytest subprocess.
        existing_pp = os.environ.get("PYTHONPATH", "")
        env_vars["PYTHONPATH"] = _EXT_ABORT_PLUGIN_DIR + (
            ":" + existing_pp if existing_pp else ""
        )

        # Clear any stale crash detection file before starting
        clear_crash_file(last_running_file)

        crashed_tests = []
        tests_to_skip = []  # Tests to deselect due to crashes
        max_retries = len(tests_list)  # At most one crash per test
        retry_count = 0

        total_tests = len(tests_list)
        print(f"\n[GPU {target_gpu}] Running {total_tests} tests from {log_name}")

        while retry_count <= max_retries:
            # Build pytest command using the utility function
            cmd = build_pytest_command(
                json_log_file,
                html_log_file,
                tests_list,
                continue_on_fail,
                reruns=3,
                deselect_tests=tests_to_skip,
            )

            _, _, _ = run_shell_command(cmd, env_vars=env_vars)

            # Check for crash
            # Use min_duration=0.0 here: if the marker exists with status=running
            # after pytest returns, we want to attribute it as a hard crash even
            # when the crash happened very quickly.
            crash_info = check_for_crash_file(last_running_file, min_duration=0.0)

            if not crash_info:
                # No crash - all remaining tests completed
                break

            # Crash detected!
            crashed_test_nodeid = crash_info["nodeid"]

            # Safety: Check if we already processed this crash
            if crashed_test_nodeid in tests_to_skip:
                print(
                    f"[WARNING] Already processed crash for {crashed_test_nodeid}, breaking"
                )
                break

            crashed_tests.append(crash_info)
            tests_to_skip.append(crashed_test_nodeid)

            print(
                f"\n[CRASH] {crashed_test_nodeid} - Re-running remaining tests "
                f"({retry_count + 1}/{max_retries})"
            )

            # Delete crash file (will be added to report later)
            clear_crash_file(last_running_file)

            retry_count += 1
            if retry_count > max_retries:
                print("[CRASH] Max retries reached")
                break

        # After all retries complete, add crashed tests to the final report
        for crash_info in crashed_tests:
            handle_abort(
                json_log_file, html_log_file, last_running_file, log_name, crash_info
            )

        # Final summary
        print(f"\n[GPU {target_gpu}] {log_name} Summary:")
        print(f"  Total: {total_tests}")
        print(f"  Crashed: {len(crashed_tests)}")

        if crashed_tests:
            print(f"\n[CRASH SUMMARY] {log_name}:")
            for crash in crashed_tests:
                print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")

            # Add to global crash list
            with GPU_LOCK:
                ALL_CRASHED_TESTS.extend(crashed_tests)

    finally:
        # CRITICAL: Always return GPU token
        with GPU_LOCK:
            gpu_tokens.append(target_gpu)
            print(f"[GPU {target_gpu}] Released GPU token")


def _create_abort_row_html(testfile, abort_info):
    """Create HTML row for abort test."""
    test_identifier = abort_info["test_name"]
    test_class = abort_info.get("test_class", "UnknownClass")

    # If test_identifier is already a full nodeid, use it; otherwise construct one
    if "::" in test_identifier and test_identifier.startswith("tests/"):
        # Already a full nodeid like "tests/image_test.py::ImageTest::testMethod"
        display_name = test_identifier
    elif "::" in test_identifier:
        # Partial nodeid like "ImageTest::testMethod" - add the file part
        display_name = f"tests/{testfile}.py::{test_identifier}"
    else:
        # Just a test method name
        display_name = f"tests/{testfile}.py::{test_identifier}"

    duration = abort_info.get("duration", 0)
    abort_time = abort_info.get("abort_time", "")
    gpu_id = abort_info.get("gpu_id", "unknown")

    # Convert duration to HH:MM:SS format matching pytest-html format
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    abort_reason = sanitize_for_json(
        abort_info.get("reason", "Test aborted or crashed.")
    )
    test_class_clean = sanitize_for_json(str(test_class))
    abort_time_clean = sanitize_for_json(str(abort_time))
    gpu_id_clean = sanitize_for_json(str(gpu_id))

    log_content = (
        f"Test aborted: {abort_reason}\n"
        f"Test Class: {test_class_clean}\n"
        f"Abort detected at: {abort_time_clean}\n"
        f"GPU ID: {gpu_id_clean}"
    )

    return f"""
                <tbody class="results-table-row">
                    <tr class="collapsible">
                        <td class="col-result">Failed</td>
                        <td class="col-name">{display_name}</td>
                        <td class="col-duration">{duration_str}</td>
                        <td class="col-links"></td>
                    </tr>
                    <tr class="extras-row">
                        <td class="extra" colspan="4">
                            <div class="extraHTML"></div>
                            <div class="logwrapper">
                                <div class="logexpander"></div>
                                <div class="log">{log_content}</div>
                            </div>
                        </td>
                    </tr>
                </tbody>"""


def _update_html_summary_counts(html_content):
    """Update test counts in HTML summary."""
    # Fix malformed run-count patterns first
    malformed_pattern = r"(\d+/\d+ test done\.)"
    if re.search(malformed_pattern, html_content):
        # Replace malformed pattern with proper format
        html_content = re.sub(malformed_pattern, "1 tests took 00:00:01.", html_content)

    # Update "X tests ran in Y" pattern (legacy format)
    count_pattern = r"(\d+) tests? ran in"
    match = re.search(count_pattern, html_content)
    if match:
        current_count = int(match.group(1))
        new_count = current_count + 1
        html_content = re.sub(count_pattern, f"{new_count} tests ran in", html_content)

    # Update "X test took" pattern (current pytest-html format)
    count_pattern2 = r"(\d+) tests? took"
    match = re.search(count_pattern2, html_content)
    if match:
        current_count = int(match.group(1))
        new_count = current_count + 1
        html_content = re.sub(count_pattern2, f"{new_count} tests took", html_content)

    # Update "X Failed" count in the summary
    failed_pattern = r"(\d+) Failed"
    match = re.search(failed_pattern, html_content)
    if match:
        current_failed = int(match.group(1))
        new_failed = current_failed + 1
        html_content = re.sub(failed_pattern, f"{new_failed} Failed", html_content)
    else:
        # If no failed tests before, enable the failed filter and update count
        html_content = html_content.replace("0 Failed,", "1 Failed,")
        html_content = html_content.replace(
            'data-test-result="failed" disabled',
            'data-test-result="failed"',
        )

    return html_content


def _update_html_json_data(
    html_content, testfile, abort_info
):  # pylint: disable=too-many-locals
    """Update JSON data in HTML file by adding crashed test info."""

    jsonblob_pattern = r'data-jsonblob="([^"]*)"'
    match = re.search(jsonblob_pattern, html_content)
    if not match:
        return html_content

    try:
        # Decode the HTML-escaped JSON
        json_str = html.unescape(match.group(1))
        existing_json = json.loads(json_str)

        # Add the abort test to the tests array
        if "tests" not in existing_json:
            existing_json["tests"] = {}

        test_name = abort_info["test_name"]
        duration = abort_info.get("duration", 0)
        abort_time = abort_info.get("abort_time", "")
        gpu_id = abort_info.get("gpu_id", "unknown")

        # Convert duration to HH:MM:SS format
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create new test entry
        test_id = f"test_{len(existing_json['tests'])}"

        # Sanitize the abort reason to remove control characters
        abort_reason = sanitize_for_json(
            abort_info.get("reason", "Test aborted or crashed.")
        )
        abort_time_clean = sanitize_for_json(str(abort_time))
        gpu_id_clean = sanitize_for_json(str(gpu_id))

        log_msg = (
            f"Test aborted: {abort_reason}\n"
            f"Abort detected at: {abort_time_clean}\n"
            f"GPU ID: {gpu_id_clean}"
        )
        new_test = {
            "testId": (
                test_name
                if ("::" in test_name and test_name.startswith("tests/"))
                else f"tests/{testfile}.py::{test_name}"
            ),
            "id": test_id,
            "log": log_msg,
            "extras": [],
            "resultsTableRow": [
                '<td class="col-result">Failed</td>',
                (
                    f'<td class="col-name">'
                    f'{test_name if ("::" in test_name and test_name.startswith("tests/")) else f"tests/{testfile}.py::{test_name}"}'  # pylint: disable=line-too-long
                    f"</td>"
                ),
                f'<td class="col-duration">{duration_str}</td>',
                '<td class="col-links"></td>',
            ],
            "tableHtml": [],
            "result": "failed",
            "collapsed": False,
        }
        existing_json["tests"][test_id] = new_test

        # Re-encode the JSON and escape for HTML
        updated_json_str = html.escape(json.dumps(existing_json))
        html_content = re.sub(
            jsonblob_pattern,
            f'data-jsonblob="{updated_json_str}"',
            html_content,
        )

    except (  # pylint: disable=broad-except
        json.JSONDecodeError,
        Exception,
    ) as ex:  # pylint: disable=broad-except
        print(f"Warning: Could not update JSON data in HTML file: {ex}")

    return html_content


def _create_new_html_file(
    html_file, testfile, abort_info
):  # pylint: disable=too-many-locals
    """Create a new HTML file for abort-only report."""
    try:
        # Create the complete HTML structure matching pytest-html format
        test_name = abort_info["test_name"]
        duration = abort_info.get("duration", 0)
        abort_time = abort_info.get("abort_time", "")
        gpu_id = abort_info.get("gpu_id", "unknown")

        # Convert duration to HH:MM:SS format as expected by pytest-html
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create JSON data for the data-container
        abort_reason = sanitize_for_json(
            abort_info.get("reason", "Test aborted or crashed.")
        )
        abort_time_clean = sanitize_for_json(str(abort_time))
        gpu_id_clean = sanitize_for_json(str(gpu_id))

        log_msg = (
            f"Test aborted: {abort_reason}\n"
            f"Abort detected at: {abort_time_clean}\n"
            f"GPU ID: {gpu_id_clean}"
        )
        json_data = {
            "environment": {
                "Python": "3.x",
                "Platform": "Linux",
                "Packages": {"pytest": "8.4.1", "pluggy": "1.6.0"},
                "Plugins": {
                    "rerunfailures": "15.1",
                    "json-report": "1.5.0",
                    "html": "4.1.1",
                    "reportlog": "0.4.0",
                    "metadata": "3.1.1",
                    "hypothesis": "6.136.6",
                },
            },
            "tests": {
                "test_0": {
                    "testId": (
                        test_name
                        if ("::" in test_name and test_name.startswith("tests/"))
                        else f"tests/{testfile}.py::{test_name}"
                    ),
                    "id": "test_0",
                    "log": log_msg,
                    "extras": [],
                    "resultsTableRow": [
                        '<td class="col-result">Failed</td>',
                        (
                            f'<td class="col-name">'
                            f'{test_name if ("::" in test_name and test_name.startswith("tests/")) else f"tests/{testfile}.py::{test_name}"}'  # pylint: disable=line-too-long
                            f"</td>"
                        ),
                        f'<td class="col-duration">{duration_str}</td>',
                        '<td class="col-links"></td>',
                    ],
                    "tableHtml": [],
                    "result": "failed",
                    "collapsed": False,
                }
            },
            "renderCollapsed": ["passed"],
            "initialSort": "result",
            "title": f"{testfile}_log.html",
        }

        # Convert JSON to HTML-escaped string for data-jsonblob attribute
        json_blob = html.escape(json.dumps(json_data))

        # Create the HTML content
        current_time_str = datetime.now().strftime("%d-%b-%Y at %H:%M:%S")
        log_content = (
            f"Test aborted: {abort_reason}\n"
            f"Abort detected at: {abort_time}\n"
            f"GPU ID: {gpu_id}"
        )

        html_content = _generate_html_template(
            {
                "testfile": testfile,
                "test_name": test_name,
                "duration_str": duration_str,
                "current_time_str": current_time_str,
                "log_content": log_content,
                "json_blob": json_blob,
            }
        )

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(html_file), exist_ok=True)

        # Write the HTML file
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Created new HTML report: {html_file}")

    except (OSError, IOError) as io_err:
        print(f"Failed to write new HTML report for {testfile}: {io_err}")
    except Exception as ex:  # pylint: disable=broad-except
        print(f"Unexpected error creating new HTML report for {testfile}: {ex}")
        traceback.print_exc()


def _generate_html_template(template_data):
    """Generate HTML template for test report."""
    testfile = template_data["testfile"]
    test_name = template_data["test_name"]
    duration_str = template_data["duration_str"]
    current_time_str = template_data["current_time_str"]
    log_content = template_data["log_content"]
    json_blob = template_data["json_blob"]
    return f"""<!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8"/>
            <title id="head-title">{testfile}_log.html</title>
            <link href="assets/style.css" rel="stylesheet" type="text/css"/>
          </head>
          <body onLoad="init()">
            <h1 id="title">{testfile}_log.html</h1>
            <p>Report generated on {current_time_str} by
               <a href="https://pypi.python.org/pypi/pytest-html">pytest-html</a> v4.1.1</p>
            <div id="environment-header">
              <h2>Environment</h2>
            </div>
            <table id="environment"></table>
            <div class="summary">
              <div class="summary__data">
                <h2>Summary</h2>
                <div class="additional-summary prefix">
                </div>
                <p class="run-count">1 tests took {duration_str}.</p>
                <p class="filter">(Un)check the boxes to filter the results.</p>
                <div class="summary__reload">
                  <div class="summary__reload__button hidden" onclick="location.reload()">
                    <div>There are still tests running. <br />Reload this page to get the latest results!</div>
                  </div>
                </div>
                <div class="summary__spacer"></div>
                <div class="controls">
                  <div class="filters">
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="failed" />
                    <span class="failed">1 Failed,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="passed" disabled/>
                    <span class="passed">0 Passed,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="skipped" disabled/>
                    <span class="skipped">0 Skipped,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="xfailed" disabled/>
                    <span class="xfailed">0 Expected failures,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="xpassed" disabled/>
                    <span class="xpassed">0 Unexpected passes,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="error" disabled/>
                    <span class="error">0 Errors,</span>
                    <input checked="true" class="filter" name="filter_checkbox" type="checkbox" data-test-result="rerun" disabled/>
                    <span class="rerun">0 Reruns</span>
                  </div>
                  <div class="collapse">
                    <button id="show_all_details">Show all details</button>&nbsp;/&nbsp;<button id="hide_all_details">Hide all details</button>
                  </div>
                </div>
              </div>
              <div class="additional-summary summary">
              </div>
              <div class="additional-summary postfix">
              </div>
            </div>
            <table id="results-table">
              <thead id="results-table-head">
                <tr>
                  <th class="sortable result initial-sort" data-column-type="result">Result</th>
                  <th class="sortable" data-column-type="name">Test</th>
                  <th class="sortable" data-column-type="duration">Duration</th>
                  <th class="sortable links" data-column-type="links">Links</th>
                </tr>
              </thead>
              <tbody class="results-table-row">
                <tr class="collapsible">
                  <td class="col-result">Failed</td>
                  <td class="col-name">{test_name}</td>
                  <td class="col-duration">{duration_str}</td>
                  <td class="col-links"></td>
                </tr>
                <tr class="extras-row">
                  <td class="extra" colspan="4">
                    <div class="extraHTML"></div>
                    <div class="logwrapper">
                      <div class="logexpander"></div>
                      <div class="log">{log_content}</div>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
            <div id="data-container" data-jsonblob="{json_blob}"></div>
            <script>
              function init() {{
                // Initialize any required functionality
              }}
            </script>
          </body>
        </html>"""


def run_parallel(tests_dict, p, c):
    """Run tests in parallel across multiple GPUs."""
    print(f"Running tests with parallelism = {p}")
    available_gpu_tokens = list(range(p))
    executor = ThreadPoolExecutor(max_workers=p)
    # walking through test modules.
    for log_name, tests_list in tests_dict.items():
        executor.submit(run_test, log_name, tests_list, available_gpu_tokens, c)
    # waiting for all modules to finish.
    executor.shutdown(wait=True)


def main(args):
    """Main function to run all test modules."""
    if getattr(args, "only_test_files", None):
        tests_dict = _tests_dict_from_only_test_files(args.only_test_files)
        print(f"only_test_files set; running {len(tests_dict)} test files:")
        for k, v in tests_dict.items():
            print(f"  - {k}: {v[0]}")
    else:
        tests_dict = collect_testmodules(args.ignore_skipfile)
    run_parallel(tests_dict, args.parallel, args.continue_on_fail)
    generate_final_report_plugin(BASE_DIR)

    # Print comprehensive final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    # pylint: disable=too-many-nested-blocks
    # Parse the final compiled report for statistics
    try:
        combined_json_file = f"{BASE_DIR}/final_compiled_report.json"
        if os.path.exists(combined_json_file):
            with open(combined_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            total_passed = 0
            total_failed = 0
            total_skipped = 0
            total_error = 0
            total_tests = 0
            crashed_tests_from_report = []
            collection_error_files = set()

            # Count collection errors (tests that failed to import/collect)
            for report in data:
                if "collectors" in report:
                    for collector in report["collectors"]:
                        if collector.get("outcome") == "failed":
                            total_error += 1
                            # Track which files have collection errors
                            collection_error_files.add(collector.get("nodeid"))

            # Detect crashes from the JSON report itself
            for report in data:
                if "tests" in report:
                    for test in report["tests"]:
                        if test.get("outcome") == "failed":
                            longrepr = str(test.get("call", {}).get("longrepr", ""))
                            # Check if this is a crash (not a regular failure)
                            if "Test aborted: Test crashed" in longrepr:
                                crashed_tests_from_report.append(
                                    {
                                        "nodeid": test.get("nodeid"),
                                        "duration": test.get("call", {}).get(
                                            "duration", 0
                                        ),
                                    }
                                )

            # Use crashes from report if ALL_CRASHED_TESTS is empty (running summary standalone)
            # Otherwise use the in-memory list from the live run
            crashes_to_use = (
                ALL_CRASHED_TESTS if ALL_CRASHED_TESTS else crashed_tests_from_report
            )
            total_crashed = len(crashes_to_use)
            crashed_nodeids = {crash["nodeid"] for crash in crashes_to_use}

            for report in data:
                if "summary" in report:
                    summary = report["summary"]
                    total_passed += summary.get("passed", 0)
                    total_skipped += summary.get("skipped", 0)
                    total_tests += summary.get("total", 0)

                    # Count failed tests, but exclude crashed tests and collection errors
                    report_failed = summary.get("failed", 0)

                    # Skip counting failures if this file had a collection error
                    file_path = report.get("root", "")
                    if any(
                        err_file in file_path for err_file in collection_error_files
                    ):
                        continue

                    # Check if any failed tests in this report are crashes
                    if report_failed > 0 and "tests" in report:
                        for test in report["tests"]:
                            if (
                                test.get("outcome") == "failed"
                                and test.get("nodeid") not in crashed_nodeids
                            ):
                                total_failed += 1
                    elif report_failed > 0:
                        # If we don't have test details, count all failed
                        total_failed += report_failed

            print(f"Total Tests:   {total_tests}")
            print(f"Passed:        {total_passed}")
            print(f"Failed:        {total_failed}")
            print(f"Skipped:       {total_skipped}")
            print(f"Errors:        {total_error}")
            print(f"Crashed:       {total_crashed}")
    except (OSError, IOError, json.JSONDecodeError) as e:
        print(f"Could not parse final report: {e}")

    # Print crashed tests list
    if ALL_CRASHED_TESTS:
        print("\n" + "-" * 70)
        print(f"CRASHED TESTS ({len(ALL_CRASHED_TESTS)}):")
        print("-" * 70)
        for crash in ALL_CRASHED_TESTS:
            print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")
    elif crashes_to_use:
        # If we detected crashes from the report (not live run)
        print("\n" + "-" * 70)
        print(f"CRASHED TESTS ({len(crashes_to_use)}):")
        print("-" * 70)
        for crash in crashes_to_use:
            print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")

    print("=" * 70)

    sys.exit(LAST_CODE)


if __name__ == "__main__":
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"

    # Archive old logs (if non-empty) and create a fresh logs directory.
    ensure_logs_dir("./logs", archive_if_nonempty=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--parallel", type=int, help="number of tests to run in parallel"
    )
    parser.add_argument(
        "-c", "--continue_on_fail", action="store_true", help="continue on failure"
    )
    parser.add_argument(
        "-s",
        "--ignore_skipfile",
        action="store_true",
        help="Ignore the test skip file and run all single GPU tests",
    )
    parser.add_argument(
        "--only-test-files",
        default="",
        help=(
            "Comma-separated list of test files to run instead of collecting all tests. "
            "Examples: 'jax/tests/linalg_test.py,jax/tests/crash_abort_test.py' or "
            "'linalg_test.py,crash_abort_test.py' (shorthand expands under jax/tests/)."
        ),
    )
    parsed_args = parser.parse_args()
    if parsed_args.continue_on_fail:
        print("continue on fail is set")
    if parsed_args.parallel is None:
        sys_gpu_count = detect_amd_gpus()
        parsed_args.parallel = sys_gpu_count
        print(f"{sys_gpu_count} GPUs detected.")

    main(parsed_args)
