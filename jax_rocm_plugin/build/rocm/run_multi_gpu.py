#!/usr/bin/env python3
# Copyright 2025 The JAX Authors.
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
Multi-GPU test runner for JAX with ROCm.
Python equivalent of [run_multi_gpu.sh]
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add the configuration directory to Python path
sys.path.insert(0, "jax_rocm_plugin/build/rocm")

try:
    from multi_gpu_tests_config import MULTI_GPU_TESTS
    from run_single_gpu import (
        check_for_crash,
        handle_abort,
        generate_final_report,
        detect_amd_gpus,
        clear_crash_file,
        build_pytest_command,
        convert_json_to_csv,
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

LOG_DIR = "./logs"
MAX_GPUS_PER_TEST = 8  # Limit for stability


def cleanup_system():
    """Clean up system resources between tests."""
    print("Cleaning up system resources...")

    # Kill any remaining pytest processes
    try:
        subprocess.run(
            ["pkill", "-f", "python.*pytest"], check=False, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Wait for cleanup
    time.sleep(5)

    # Clear shared memory if possible
    try:
        subprocess.run(
            ["find", "/dev/shm", "-name", "*jax*", "-delete"],
            check=False,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Additional wait
    time.sleep(3)


def check_system_resources():
    """Check available system resources."""
    try:
        # Check memory
        result = subprocess.run(
            ["free", "-g"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "Mem:" in line:
                parts = line.split()
                available = int(parts[6])  # Available memory
                if available < 10:
                    print(f"WARNING: Low memory available: {available}GB")
                    return False
        return True
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return True  # Continue if check fails


# pylint: disable=unused-argument
def get_deselected_tests(test_name):
    """filter out listed test for a given test_name."""
    return []


# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
def run_multi_gpu_test(
    test_file, gpu_count, continue_on_fail, max_gpus=None, ignore_skipfile=False
):
    """Run a single multi-GPU test file with crash recovery.

    Tests are run ONE AT A TIME for reliable crash recovery.
    If a test crashes, we simply move to the next test.
    """
    if max_gpus and gpu_count > max_gpus:
        gpu_count = max_gpus
        print(f"Limiting GPU count to {max_gpus} for stability")

    # Create GPU list (0,1,2,3 for 4 GPUs)
    gpu_list = ",".join(str(i) for i in range(gpu_count))

    # Extract test name for logging
    test_name = Path(test_file).stem

    # Setup file paths
    abs_json_log_file = os.path.abspath(f"{LOG_DIR}/multi_gpu_{test_name}_log.json")
    abs_html_log_file = os.path.abspath(f"{LOG_DIR}/multi_gpu_{test_name}_log.html")
    abs_last_running_file = os.path.abspath(
        f"{LOG_DIR}/multi_gpu_{test_name}_last_running.json"
    )

    print(f"=== Starting multi-GPU test: {test_file} ===")
    print(f"GPUs: {gpu_list} (count: {gpu_count})")
    print(f"Timestamp: {datetime.now()}")

    # Check resources before test
    if not check_system_resources():
        print("Waiting for system resources...")
        time.sleep(30)

    # Clear any stale crash detection file
    clear_crash_file(abs_last_running_file)

    # Environment setup
    env = os.environ.copy()
    env.update(
        {
            "HIP_VISIBLE_DEVICES": gpu_list,
            "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
        }
    )

    # Get deselected tests (permanent skips for this test file)
    permanent_skips = []
    if not ignore_skipfile:
        permanent_skips = get_deselected_tests(test_name)

    # First, collect all tests in this file
    print(f"Collecting tests from {test_file}...")
    collect_cmd = [
        "python3",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        f"./jax/{test_file}",
        *permanent_skips,
    ]

    try:
        result = subprocess.run(
            collect_cmd, env=env, capture_output=True, text=True, check=False
        )
        # Parse collected test IDs from output
        test_nodeids = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith(test_file) or line.startswith(f"./{test_file}"):
                test_nodeids.append(f"./jax/{line}")

        if not test_nodeids:
            print(f"[WARNING] No tests collected from {test_file}")
            return (0, [])

        print(f"Collected {len(test_nodeids)} tests")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] Failed to collect tests: {e}")
        return (1, [])

    # Run all tests together, re-running with --deselect on crash
    crashed_tests = []
    tests_to_skip = []
    max_retries = len(test_nodeids)
    retry_count = 0
    total_tests = len(test_nodeids)

    while retry_count <= max_retries:
        # Clear crash file before each run
        clear_crash_file(abs_last_running_file)

        # Build pytest command with deselect for crashed tests
        cmd = build_pytest_command(
            abs_json_log_file,
            abs_html_log_file,
            test_nodeids,
            continue_on_fail,
            deselect_tests=tests_to_skip,
        )

        print(f"Running: {' '.join(cmd)}")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per batch
                check=False,
            )

            duration = time.time() - start_time
            return_code = result.returncode

            print(f"Tests completed in {duration:.2f}s with exit code: {return_code}")

            if result.stdout:
                print("STDOUT:", result.stdout[-500:])
            if result.stderr:
                print("STDERR:", result.stderr[-500:])

            # Check for crash
            crash_info = check_for_crash(abs_last_running_file)

            if not crash_info:
                # No crash - all remaining tests completed
                break

            # Crash detected!
            crashed_test_nodeid = crash_info["nodeid"]

            # Safety: prevent infinite loop
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
            clear_crash_file(abs_last_running_file)

            retry_count += 1
            if retry_count > max_retries:
                print("[CRASH] Max retries reached")
                break

        except subprocess.TimeoutExpired:
            print("ERROR: Tests timed out after 1 hour")
            break
        except (subprocess.SubprocessError, OSError) as os_e:
            print(f"ERROR: Exception running tests: {os_e}")
            break
        finally:
            cleanup_system()

    # After all retries complete, add crashed tests to the final report
    for crash_info in crashed_tests:
        handle_abort(
            abs_json_log_file,
            abs_html_log_file,
            abs_last_running_file,
            f"multi_gpu_{test_name}",
            crash_info,
        )

    # Final summary
    print(f"\n=== {test_file} Summary ===")
    print(f"Total: {total_tests}")
    print(f"Crashed: {len(crashed_tests)}")

    if crashed_tests:
        print(f"\n[CRASH SUMMARY] {test_name}:")
        for crash in crashed_tests:
            print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")

    # Return exit code and crashed tests list
    has_failures = len(crashed_tests) > 0
    exit_code = 1 if has_failures else 0
    return exit_code, crashed_tests


def main():
    """Main function to run all multi-GPU tests."""
    parser = argparse.ArgumentParser(description="Run multi-GPU JAX tests")
    parser.add_argument(
        "--gpu-count", type=int, help="Number of GPUs to use (default: auto-detect)"
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=MAX_GPUS_PER_TEST,
        help=f"Maximum GPUs per test (default: {MAX_GPUS_PER_TEST})",
    )
    parser.add_argument(
        "--test-filter", type=str, help="Run only tests containing this string"
    )
    parser.add_argument(
        "-c", "--continue_on_fail", action="store_true", help="continue on failure"
    )
    parser.add_argument(
        "-s",
        "--ignore_skipfile",
        action="store_true",
        help="Ignore the test skip file and run all mutli GPU tests",
    )
    args = parser.parse_args()

    # Detect GPU count if not specified
    if args.gpu_count is None:
        args.gpu_count = detect_amd_gpus()
        print(f"Detected {args.gpu_count} AMD GPUs")

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Filter tests if requested
    tests_to_run = MULTI_GPU_TESTS
    if args.test_filter:
        tests_to_run = {test for test in MULTI_GPU_TESTS if args.test_filter in test}
        print(
            f"Filtered to {len(tests_to_run)} tests containing " f"'{args.test_filter}'"
        )

    print(
        f"Running {len(tests_to_run)} multi-GPU tests with up to "
        f"{args.max_gpus} GPUs each"
    )

    # Run tests sequentially
    failed_tests = []
    passed_tests = []
    all_crashed_tests = []  # Collect all crashed tests

    for i, test_file in enumerate(sorted(tests_to_run), 1):
        print(f"\n[{i}/{len(tests_to_run)}] Running {test_file}")

        try:
            exit_code, crashed_tests = run_multi_gpu_test(
                test_file,
                args.gpu_count,
                args.continue_on_fail,
                max_gpus=args.max_gpus,
                ignore_skipfile=args.ignore_skipfile,
            )

            # Collect crashed tests
            all_crashed_tests.extend(crashed_tests)

            if exit_code == 0:
                passed_tests.append(test_file)
            else:
                failed_tests.append((test_file, exit_code))
                if not args.continue_on_fail:
                    print("fail-fast: stopping after first failure")
                    break

        except KeyboardInterrupt:
            print(f"\nInterrupted during {test_file}")
            break
        except (subprocess.SubprocessError, OSError) as os_e:
            print(f"ERROR: Exception with {test_file}: {os_e}")
            failed_tests.append((test_file, -1))

    # Generate final report (reuse from run_single_gpu.py)
    try:
        generate_final_report()
        print("Final HTML and JSON reports generated")

        # Generate CSV report for multi-GPU tests
        combined_json_file = f"{LOG_DIR}/final_compiled_report.json"
        combined_csv_file = f"{LOG_DIR}/final_compiled_report.csv"
        convert_json_to_csv(combined_json_file, combined_csv_file)
        print("Final CSV report generated")
    except (ImportError, OSError, ValueError) as excp:
        print(f"Warning: Could not generate final report: {excp}")

    # Print comprehensive final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    # pylint: disable=too-many-nested-blocks,import-outside-toplevel
    # Parse the final compiled report for statistics
    try:
        combined_json_file = f"{LOG_DIR}/final_compiled_report.json"
        if os.path.exists(combined_json_file):
            import json as json_module

            with open(combined_json_file, "r", encoding="utf-8") as f:
                data = json_module.load(f)

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

            # Use crashes from report if all_crashed_tests is empty (running summary standalone)
            # Otherwise use the in-memory list from the live run
            crashes_to_use = (
                all_crashed_tests if all_crashed_tests else crashed_tests_from_report
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
    except (OSError, IOError, Exception) as e:  # pylint: disable=broad-except
        print(f"Could not parse final report: {e}")

    # Print crashed tests list
    if all_crashed_tests:
        print("\n" + "-" * 70)
        print(f"CRASHED TESTS ({len(all_crashed_tests)}):")
        print("-" * 70)
        for crash in all_crashed_tests:
            print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")
    elif crashes_to_use:
        # If we detected crashes from the report (not live run)
        print("\n" + "-" * 70)
        print(f"CRASHED TESTS ({len(crashes_to_use)}):")
        print("-" * 70)
        for crash in crashes_to_use:
            print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")

    # Print failed test files
    if failed_tests:
        print("\n" + "-" * 70)
        print(f"FAILED TEST FILES ({len(failed_tests)}):")
        print("-" * 70)
        for test_file, exit_code in failed_tests:
            print(f"  - {test_file} (exit code: {exit_code})")

    print("=" * 70)

    # Exit with failure if any tests failed and not continue_on_fail
    if args.continue_on_fail:
        sys.exit(0)
    sys.exit(1 if (failed_tests or all_crashed_tests) else 0)


if __name__ == "__main__":
    # Set ROCm environment
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"

    # Just ensure logs directory exists (don't archive, bec we are 
    # going to use logs that run_single_gpu.py creates)
    logs_dir = os.path.abspath("./logs")
    os.makedirs(logs_dir, exist_ok=True)

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)

