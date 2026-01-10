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


# pylint: disable=line-too-long
def get_deselected_tests(test_name):
    """filter out listed test for a given test_name."""
    tests_dict = {
        "export_harnesses_multi_platform_test": [
            "--deselect=tests/export_harnesses_multi_platform_test.py::PrimitiveTest::test_prim_tridiagonal_solve_shape_float32_3_",
            "--deselect=tests/export_harnesses_multi_platform_test.py::PrimitiveTest::test_prim_tridiagonal_solve_shape_float64_3_",
        ],
        "linalg_sharding_test": [
            "--deselect=tests/linalg_sharding_test.py::LinalgShardingTest::test_batch_axis_sharding_jvp13",
            "--deselect=tests/linalg_sharding_test.py::LinalgShardingTest::test_batch_axis_sharding_vjp11",
        ],
        "multi_device_test": [
            "--deselect=tests/multi_device_test.py::MultiDeviceTest::test_lax_full_like_efficient",
        ],
        "pgle_test": [
            "--deselect=tests/pgle_test.py::PgleTest::testAutoPgle",
            "--deselect=tests/pgle_test.py::PgleTest::testAutoPgleWithCommandBuffers0",
            "--deselect=tests/pgle_test.py::PgleTest::testAutoPgleWithCommandBuffers1",
            "--deselect=tests/pgle_test.py::PgleTest::testAutoPgleWithPersistentCache",
            "--deselect=tests/pgle_test.py::PgleTest::testPGLEProfilerGetFDOProfile",
            "--deselect=tests/pgle_test.py::PgleTest::testPGLEProfilerGetFDOProfileLarge",
        ],
        "pjit_test": [
            "--deselect=tests/pjit_test.py::ShardingInTypesTest::test_sparse_linalg_cg_indexing",
        ],
        "shard_map_test": [
            "--deselect=tests/shard_map_test.py::ShardMapTest::test_psend_precv_basic_two_gpus",
            "--deselect=tests/shard_map_test.py::ShardMapTest::test_psend_precv_basic_with_deadlock_cycle",
            "--deselect=tests/shard_map_test.py::ShardMapTest::test_psend_precv_basic_with_duplicate_source_target_pairs",
            "--deselect=tests/shard_map_test.py::ShardMapTest::test_psend_precv_basic_with_no_deadlock_cycle",
            "--deselect=tests/shard_map_test.py::ShardMapTest::test_psend_precv_basic_with_non_matching_source_target_pairs",
            "--deselect=tests/shard_map_test.py::ShardMapTest::test_psend_precv_reverse_two_gpus",
        ],
    }
    return tests_dict[test_name] if test_name in tests_dict else []


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
            return 0

        print(f"Collected {len(test_nodeids)} tests")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] Failed to collect tests: {e}")
        return 1

    # Run tests ONE AT A TIME
    crashed_tests = []
    passed_count = 0
    failed_count = 0
    total_tests = len(test_nodeids)

    for test_idx, test_nodeid in enumerate(test_nodeids, 1):
        print(f"\n[{test_idx}/{total_tests}] Running: {test_nodeid}")

        # Clear crash file before each test
        clear_crash_file(abs_last_running_file)

        # Build pytest command for single test
        cmd = build_pytest_command(
            abs_json_log_file, abs_html_log_file, test_nodeid, continue_on_fail
        )

        print(f"Running: {' '.join(cmd)}")

        start_time = time.time()
        try:
            # Run the single test
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per test
                check=False,
            )

            duration = time.time() - start_time
            return_code = result.returncode

            print(f"Test completed in {duration:.2f}s with exit code: {return_code}")

            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars

            # Check for crash
            crash_info = check_for_crash(abs_last_running_file)

            if crash_info:
                # Crash detected!
                crashed_test_nodeid = crash_info["nodeid"]
                crashed_tests.append(crash_info)
                failed_count += 1

                print(
                    f"[CRASH] {crashed_test_nodeid} crashed after {crash_info['duration']:.1f}s"
                )

                # Process the crash
                handle_abort(
                    abs_json_log_file,
                    abs_html_log_file,
                    abs_last_running_file,
                    f"multi_gpu_{test_name}",
                )

                # Continue to next test
                continue

            # Test completed normally
            if return_code == 0:
                passed_count += 1
            else:
                failed_count += 1
                if not continue_on_fail:
                    print("Stopping on first failure (no --continue_on_fail)")
                    break

        except subprocess.TimeoutExpired:
            print(f"ERROR: Test {test_nodeid} timed out after 1 hour")
            failed_count += 1
            if not continue_on_fail:
                break
        except (subprocess.SubprocessError, OSError) as os_e:
            print(f"ERROR: Exception running test {test_nodeid}: {os_e}")
            failed_count += 1
            if not continue_on_fail:
                break
        finally:
            cleanup_system()

    # Final summary
    print(f"\n=== {test_file} Summary ===")
    print(f"Total: {total_tests}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Crashed: {len(crashed_tests)}")

    if crashed_tests:
        print(f"\n[CRASH SUMMARY] {test_name}:")
        for crash in crashed_tests:
            print(f"  - {crash['nodeid']} ({crash['duration']:.1f}s)")

    # Return exit code: 0 if all passed, 1 otherwise
    return 0 if failed_count == 0 and len(crashed_tests) == 0 else 1


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

    for i, test_file in enumerate(sorted(tests_to_run), 1):
        print(f"\n[{i}/{len(tests_to_run)}] Running {test_file}")

        try:
            exit_code = run_multi_gpu_test(
                test_file,
                args.gpu_count,
                args.continue_on_fail,
                max_gpus=args.max_gpus,
                ignore_skipfile=args.ignore_skipfile,
            )

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

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for test_file, exit_code in failed_tests:
            print(f"  {test_file} (exit code: {exit_code})")

    # Generate final report (reuse from run_single_gpu.py)
    try:
        generate_final_report()
        print("Final HTML report generated")
    except (ImportError, OSError, ValueError) as excp:
        print(f"Warning: Could not generate final report: {excp}")

    # Exit with failure if any tests failed and not continue_on_fail
    if args.continue_on_fail:
        print("continue on fail is set")
        sys.exit(0)
    sys.exit(1 if failed_tests else 0)


if __name__ == "__main__":
    # Set ROCm environment
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
