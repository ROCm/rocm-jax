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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict

# Add the configuration directory to Python path
sys.path.insert(0, "jax_rocm_plugin/build/rocm")

GPU_LOCK = threading.Lock()
LAST_CODE = 0
BASE_DIR = "./logs"


def detect_amd_gpus():
    """Detect number of AMD/ATI GPUs using rocm-smi."""
    try:
        cmd = [
            "bash",
            "-c",
            (
                "rocm-smi | grep -E '^Device' -A 1000 | "
                "awk '$1 ~ /^[0-9]+$/ {count++} END {print count}'"
            ),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        print("Warning: Could not detect GPUs using rocm-smi, defaulting to 8")
        return 8


def clear_crash_file(crash_file_path):
    """Clear crash detection file if it exists.

    Args:
        crash_file_path: Path to crash detection file

    Returns:
        True if file was removed or didn't exist, False if removal failed
    """
    if os.path.exists(crash_file_path):
        try:
            os.remove(crash_file_path)
            print(f"[DEBUG] Removed crash file: {crash_file_path}")
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[WARNING] Could not remove crash file: {e}")
            return False
    return True


def build_pytest_command(
    json_log_file, html_log_file, test_nodeid, continue_on_fail=False, reruns=3
):
    """Build pytest command for running a single test.

    Args:
        json_log_file: Path to JSON report file
        html_log_file: Path to HTML report file
        test_nodeid: Test node ID to run
        continue_on_fail: Whether to continue on failure (don't use -x flag)
        reruns: Number of times to rerun failed tests

    Returns:
        List of command arguments
    """
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
        test_nodeid,  # Single test
    ]

    if not continue_on_fail:
        # Add -x flag to stop on first failure (before -v)
        cmd.insert(-2, "-x")

    return cmd


def extract_test_name(path: str) -> str:
    """Return the base filename (without .py and without test suffixes)."""
    return os.path.splitext(os.path.basename(path.split("::")[0]))[0]


def combine_json_reports():
    """Combine all individual JSON test reports into a single report."""
    all_json_files = [f for f in os.listdir(BASE_DIR) if f.endswith("_log.json")]
    combined_data = []
    for json_file in all_json_files:
        with open(os.path.join(BASE_DIR, json_file), "r", encoding="utf-8") as infile:
            data = json.load(infile)
            combined_data.append(data)
    combined_json_file = f"{BASE_DIR}/final_compiled_report.json"
    with open(combined_json_file, "w", encoding="utf-8") as outfile:
        json.dump(combined_data, outfile, indent=4)


def generate_final_report(shell=False, env_vars=None):
    """Generate final HTML and JSON reports by merging individual test reports."""
    if env_vars is None:
        env_vars = {}
    env = os.environ.copy()
    env.update(env_vars)

    # First, try to merge HTML files
    cmd = [
        "pytest_html_merger",
        "-i",
        f"{BASE_DIR}",
        "-o",
        f"{BASE_DIR}/final_compiled_report.html",
    ]
    result = subprocess.run(
        cmd, shell=shell, capture_output=True, env=env, check=False, timeout=300
    )
    if result.returncode != 0:
        print(f"FAILED - {' '.join(cmd)}")
        print(result.stderr.decode())
        print("HTML merger failed, but continuing with JSON report generation...")

    # Generate json reports.
    combine_json_reports()


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


def collect_testmodules(ignore_skipfile=True):
    """Collect all test modules, excluding multi-GPU tests."""

    # copy to jax as it's node ids for pytest.
    shutil.copy("ci/pytest_drop_test_list.ini", "jax/")

    # jax/pytest_drop_test_list has ingore list for multi-gpu
    # append skip list as -deselect
    if not ignore_skipfile:
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

    print(f"collect_testmodules: cmd={pytest_cmd}")

    # run pytest collection and save in file collected_tests.txt
    with open("collected_tests.txt", "w", encoding="utf-8") as f:
        try:
            subprocess.run(pytest_cmd, stdout=f, check=True)
        except subprocess.CalledProcessError as e:
            print(f"command failed: {e}")

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

    Tests are run ONE AT A TIME to enable proper crash recovery.
    If a test crashes, we simply move to the next test.
    """
    global LAST_CODE  # pylint: disable=global-statement

    # Acquire GPU token
    with GPU_LOCK:
        if LAST_CODE != 0:
            return
        target_gpu = gpu_tokens.pop()

    try:
        # Everything inside try block to ensure GPU token is always returned
        env_vars = {
            "HIP_VISIBLE_DEVICES": str(target_gpu),
            "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
        }

        json_log_file = f"{BASE_DIR}/{log_name}_log.json"
        html_log_file = f"{BASE_DIR}/{log_name}_log.html"
        last_running_file = f"{BASE_DIR}/{log_name}_last_running.json"

        # Clear any stale crash detection file before starting
        clear_crash_file(last_running_file)

        crashed_tests = []
        passed_count = 0
        failed_count = 0
        total_tests = len(tests_list)

        print(f"\n[GPU {target_gpu}] Running {total_tests} tests from {log_name}")

        # Run tests ONE AT A TIME for easy crash recovery
        for test_idx, test_nodeid in enumerate(tests_list, 1):
            # Check if we should stop
            with GPU_LOCK:
                if LAST_CODE != 0 and not continue_on_fail:
                    print(f"[GPU {target_gpu}] Stopping due to previous failure")
                    break

            print(
                f"\n[GPU {target_gpu}] [{test_idx}/{total_tests}] Running: {test_nodeid}"
            )

            # Ensure crash detection file is cleared before each test
            clear_crash_file(last_running_file)

            # Build pytest command for single test
            cmd = build_pytest_command(
                json_log_file, html_log_file, test_nodeid, continue_on_fail
            )

            return_code, _, _ = run_shell_command(cmd, env_vars=env_vars)

            # Check for crash
            crash_info = check_for_crash(last_running_file)

            if crash_info:
                # Crash detected!
                crashed_test_nodeid = crash_info["nodeid"]
                crashed_tests.append(crash_info)
                failed_count += 1

                print(
                    f"[CRASH] {crashed_test_nodeid} crashed after {crash_info['duration']:.1f}s"
                )

                # Process the crash and add to reports
                handle_abort(json_log_file, html_log_file, last_running_file, log_name)

                # Continue to next test (crash file is deleted in handle_abort)
                continue

            # Test completed normally
            if return_code == 0:
                passed_count += 1
            else:
                failed_count += 1
                if not continue_on_fail:
                    with GPU_LOCK:
                        if LAST_CODE == 0:
                            LAST_CODE = return_code
                    break

        # Final summary
        print(f"\n[GPU {target_gpu}] {log_name} Summary:")
        print(f"  Total: {total_tests}")
        print(f"  Passed: {passed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Crashed: {len(crashed_tests)}")

        if crashed_tests:
            print(f"\n[CRASH SUMMARY] {log_name}:")
            for crash in crashed_tests:
                print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")

    finally:
        # CRITICAL: Always return GPU token, even on exception or early exit
        with GPU_LOCK:
            gpu_tokens.append(target_gpu)
            print(f"[GPU {target_gpu}] Released GPU token")


# pylint: disable=too-many-branches
def append_abort_to_json(json_file, testfile, abort_info):
    # pylint: disable=too-many-locals
    """Append abort info to JSON report in pytest format."""
    # Use the test_name which could be a full nodeid or just a test name
    test_identifier = abort_info["test_name"]
    test_class = abort_info.get("test_class", "UnknownClass")

    # If test_identifier is already a full nodeid, use it; otherwise construct one
    if "::" in test_identifier and test_identifier.startswith("tests/"):
        # Already a full nodeid like "tests/image_test.py::ImageTest::testMethod"
        test_nodeid = test_identifier
    elif "::" in test_identifier:
        # Partial nodeid like "ImageTest::testMethod" - add the file part
        test_nodeid = f"tests/{testfile}.py::{test_identifier}"
    else:
        # Just a test method name
        test_nodeid = f"tests/{testfile}.py::{test_identifier}"

    abort_longrepr = (
        f"Test aborted: {abort_info.get('reason', 'Unknown abort reason')}\n"
        f"Test Class: {test_class}\n"
        f"Abort detected at: {abort_info.get('abort_time', '')}\n"
        f"GPU ID: {abort_info.get('gpu_id', 'unknown')}"
    )

    abort_test = {
        "nodeid": test_nodeid,
        "lineno": 1,
        "outcome": "failed",
        "keywords": [abort_info["test_name"], testfile, "abort", test_class, ""],
        "setup": {"duration": 0.0, "outcome": "passed"},
        "call": {
            "duration": abort_info.get("duration", 0),
            "outcome": "failed",
            "longrepr": abort_longrepr,
        },
        "teardown": {"duration": 0.0, "outcome": "skipped"},
    }

    try:
        # Check if JSON file already exists (normal test run completed)
        if os.path.exists(json_file):
            # File exists - read existing data and append the aborted test
            with open(json_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            # Add the abort test to existing tests
            if "tests" not in report_data:
                report_data["tests"] = []
            report_data["tests"].append(abort_test)

            # Update summary counts
            if "summary" in report_data:
                summary = report_data["summary"]
                summary["failed"] = summary.get("failed", 0) + 1
                summary["total"] = summary.get("total", 0) + 1
                summary["collected"] = summary.get("collected", 0) + 1
                if "unskipped_total" in summary:
                    summary["unskipped_total"] = summary["unskipped_total"] + 1

            # Update exit code to indicate failure
            report_data["exitcode"] = 1

            print(f"Appended abort test to existing JSON report: {json_file}")
        else:
            # File doesn't exist - create complete pytest JSON report structure
            current_time = datetime.now().timestamp()
            report_data = {
                "created": current_time,
                "duration": abort_info.get("duration", 0),
                "exitcode": 1,  # Non-zero exit code for failure
                "root": "/rocm-jax/jax",
                "environment": {},
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "collected": 1,
                    "unskipped_total": 1,
                },
                "collectors": [
                    {
                        "nodeid": "",
                        "outcome": "failed",
                        "result": [
                            {"nodeid": f"tests/{testfile}.py", "type": "Module"}
                        ],
                    }
                ],
                "tests": [abort_test],
            }
            print(f"Created new JSON report with abort test: {json_file}")

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        # Write the file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    except (OSError, IOError) as io_err:
        print(f"Failed to write JSON report for {testfile}: {io_err}")
    except json.JSONDecodeError as json_err:
        print(f"Failed to parse existing JSON report for {testfile}: {json_err}")
        print("Creating new JSON file instead...")
        # Try creating a new file structure with just the abort test
        try:
            current_time = datetime.now().timestamp()
            new_report_data = {
                "created": current_time,
                "duration": abort_info.get("duration", 0),
                "exitcode": 1,
                "root": "/rocm-jax/jax",
                "environment": {},
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "collected": 1,
                    "unskipped_total": 1,
                },
                "collectors": [
                    {
                        "nodeid": "",
                        "outcome": "failed",
                        "result": [
                            {"nodeid": f"tests/{testfile}.py", "type": "Module"}
                        ],
                    }
                ],
                "tests": [abort_test],
            }
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(new_report_data, f, indent=2)
        except (OSError, IOError) as io_e:
            print(f"Failed to create new JSON report for {testfile}: {io_e}")


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

    abort_reason = abort_info.get("reason", "Test aborted or crashed.")
    log_content = (
        f"Test aborted: {abort_reason}<br/>"
        f"Test Class: {test_class}<br/>"
        f"Abort detected at: {abort_time}<br/>"
        f"GPU ID: {gpu_id}"
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
    """Update JSON data in HTML file."""
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
        log_msg = (
            f"Test aborted: {abort_info.get('reason', 'Test aborted or crashed.')}\\n"
            f"Abort detected at: {abort_time}\\n"
            f"GPU ID: {gpu_id}"
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


# pylint: disable=too-many-branches
def append_abort_to_html(html_file, testfile, abort_info):
    """Generate or append abort info to pytest-html format HTML report."""
    try:
        # Check if HTML file already exists (normal test run completed)
        if os.path.exists(html_file):
            # File exists - read and append abort test row to existing HTML
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Create abort test row HTML
            abort_row = _create_abort_row_html(testfile, abort_info)

            # Insert the abort row before the closing </table> tag
            if "</table>" in html_content:
                # Find the results-table specifically, not the environment table
                results_table_start = html_content.find('<table id="results-table">')
                results_table_end = html_content.find("</table>", results_table_start)
                if results_table_end != -1:
                    # Insert before the specific results table closing tag
                    html_content = (
                        html_content[:results_table_end]
                        + f"{abort_row}\n    "
                        + html_content[results_table_end:]
                    )
                else:
                    print(
                        f"Warning: Could not find results-table closing tag in {html_file}"
                    )
                    _create_new_html_file(html_file, testfile, abort_info)
                    return

                # Update test counts and JSON data
                html_content = _update_html_summary_counts(html_content)
                html_content = _update_html_json_data(
                    html_content, testfile, abort_info
                )

                # Ensure the reload button has the hidden class
                html_content = re.sub(
                    r'class="summary__reload__button\s*"',
                    'class="summary__reload__button hidden"',
                    html_content,
                )

                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html_content)

                print(f"Appended abort test to existing HTML report: {html_file}")
            else:
                print(
                    f"Warning: Could not find </table> tag in existing HTML file {html_file}"
                )
                # Fall back to creating new file
                _create_new_html_file(html_file, testfile, abort_info)
        else:
            # File doesn't exist - create complete new HTML file
            _create_new_html_file(html_file, testfile, abort_info)

    except (OSError, IOError) as io_err:
        print(f"Failed to read/write HTML report for {testfile}: {io_err}")
    except (json.JSONDecodeError, UnicodeDecodeError) as parse_err:
        print(f"Failed to parse existing HTML report for {testfile}: {parse_err}")
        print("Creating new HTML file instead...")
        _create_new_html_file(html_file, testfile, abort_info)


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
        abort_reason = abort_info.get("reason", "Test aborted or crashed.")
        log_msg = (
            f"Test aborted: {abort_reason}\\n"
            f"Abort detected at: {abort_time}\\n"
            f"GPU ID: {gpu_id}"
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
            f"Test aborted: {abort_reason}<br/>"
            f"Abort detected at: {abort_time}<br/>"
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


def check_for_crash(last_running_file):
    """Check if a crash occurred and return crash info.

    Returns:
        dict with crash info if crash detected, None otherwise

    A crash is detected when:
    1. The last_running file exists after pytest completes
    2. The file has valid JSON with test information
    3. The test was marked as "running" but never completed
    """
    if not os.path.exists(last_running_file):
        # File doesn't exist = no crash (test completed normally)
        return None

    try:
        with open(last_running_file, "r", encoding="utf-8") as f:
            crash_data = json.load(f)

        # Verify this is a valid crash detection file
        if crash_data.get("status") != "running":
            print(
                f"[DEBUG] Crash file has status: {crash_data.get('status')}, not 'running'"
            )
            return None

        start_time = datetime.fromisoformat(crash_data["start_time"])
        duration = (datetime.now() - start_time).total_seconds()

        # Sanity check: if duration is very short, might be false positive
        # (file from conftest.py might not have been cleared yet)
        if duration < 0.1:
            print(f"[DEBUG] Crash detection skipped - duration too short: {duration}s")
            return None

        # Use nodeid if available
        test_identifier = crash_data.get(
            "nodeid", crash_data.get("test_name", "unknown_test")
        )

        # Extract test class name
        test_class = "UnknownClass"
        if "::" in test_identifier:
            parts = test_identifier.split("::")
            if len(parts) >= 3:
                test_class = parts[1]
            elif len(parts) == 2:
                test_class = parts[1]

        return {
            "test_name": test_identifier,
            "test_class": test_class,
            "nodeid": crash_data.get("nodeid", test_identifier),
            "reason": "Test crashed (segfault, abort, or fatal error)",
            "crash_time": datetime.now().isoformat(),
            "duration": duration,
            "gpu_id": crash_data.get("gpu_id", "unknown"),
            "pid": crash_data.get("pid", "unknown"),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as ex:
        print(f"[WARNING] Invalid crash file {last_running_file}: {ex}")
        # Clean up invalid file
        try:
            os.remove(last_running_file)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return None
    except Exception as ex:  # pylint: disable=broad-except
        print(f"[ERROR] Error reading crash file {last_running_file}: {ex}")
        return None


def handle_abort(json_file, html_file, last_running_file, testfile):
    """Handle crash detection and append info to reports.

    CRITICAL: This function MUST delete the crash detection file to prevent
    infinite loops, even if processing fails.
    """
    crash_info = check_for_crash(last_running_file)

    # CRITICAL: Delete crash file IMMEDIATELY to prevent infinite loop
    # Do this before any processing that might fail
    if os.path.exists(last_running_file):
        try:
            os.remove(last_running_file)
            print(f"[DEBUG] Deleted crash file: {last_running_file}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[ERROR] Could not delete crash file {last_running_file}: {e}")
            # Continue anyway - we got the crash info already

    if not crash_info:
        return False

    try:
        append_abort_to_json(json_file, testfile, crash_info)
        append_abort_to_html(html_file, testfile, crash_info)
        return True
    except Exception as ex:  # pylint: disable=broad-except
        print(f"[ERROR] Error processing crash: {ex}")
        traceback.print_exc()
        return False


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
    tests_dict = collect_testmodules(args.ignore_skipfile)
    run_parallel(tests_dict, args.parallel, args.continue_on_fail)
    generate_final_report()
    sys.exit(LAST_CODE)


if __name__ == "__main__":
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"
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
    parsed_args = parser.parse_args()
    if parsed_args.continue_on_fail:
        print("continue on fail is set")
    if parsed_args.parallel is None:
        sys_gpu_count = detect_amd_gpus()
        parsed_args.parallel = sys_gpu_count
        print(f"{sys_gpu_count} GPUs detected.")

    main(parsed_args)
