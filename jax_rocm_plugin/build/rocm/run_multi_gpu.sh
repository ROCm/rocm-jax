#!/usr/bin/env bash
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

#!/usr/bin/env bash

set -euxo pipefail

LOG_DIR="./logs"

# --------------------------------------------------------------------------------
# Function to detect number of AMD/ATI GPUs using lspci.
# --------------------------------------------------------------------------------
detect_amd_gpus() {
    # Make sure lspci is installed.
    if ! command -v lspci &>/dev/null; then
        echo "Error: lspci command not found. Aborting."
        exit 1
    fi
    # Count AMD/ATI GPU controllers.
    local count
    count=$(lspci | grep -c 'controller.*AMD/ATI')
    echo "$count"
}

# --------------------------------------------------------------------------------
# Function to run tests with specified GPUs.
# --------------------------------------------------------------------------------
run_tests() {
    local gpu_devices="$1"

    echo "Running tests on GPUs: $gpu_devices"
    export HIP_VISIBLE_DEVICES="$gpu_devices"

    # Ensure python3 is available.
    if ! command -v python3 &>/dev/null; then
        echo "Error: Python3 is not available. Aborting."
        exit 1
    fi

    # Create the log directory if it doesn't exist.
    mkdir -p "$LOG_DIR"

    # Multi-GPU test files - load from shared configuration
    echo "Loading multi-GPU tests from configuration..."
    
    # Set the path to the configuration file
    CONFIG_PATH="jax_rocm_plugin/build/rocm/multi_gpu_tests_config.py"
    
    # Check if python3 and the config file are available
    if ! python3 -c "import sys; sys.path.insert(0, 'jax_rocm_plugin/build/rocm'); import multi_gpu_tests_config" 2>/dev/null; then
        echo "Error: multi_gpu_tests_config.py not found in jax_rocm_plugin/build/rocm/ or not importable. Aborting."
        exit 1
    fi
    
    # Load the multi-GPU tests from the configuration file
    mapfile -t MULTI_GPU_TESTS < <(python3 "$CONFIG_PATH" --list)

    # Run each multi-GPU test
    for test_file in "${MULTI_GPU_TESTS[@]}"; do
        test_name=$(basename "$test_file" .py)
        echo "Running multi-GPU test: $test_file"
        
        python3 -m pytest \
            --html="${LOG_DIR}/multi_gpu_${test_name}_log.html" \
            --json-report \
            --json-report-file="${LOG_DIR}/multi_gpu_${test_name}_log.json" \
            --reruns 3 \
            "./jax/$test_file"
    done

    # Merge individual HTML reports into one.
    python3 -m pytest_html_merger \
        -i "$LOG_DIR" \
        -o "${LOG_DIR}/final_compiled_report.html"
}

# --------------------------------------------------------------------------------
# Main entry point.
# --------------------------------------------------------------------------------
main() {
    # Detect number of AMD/ATI GPUs.
    local gpu_count
    gpu_count=$(detect_amd_gpus)
    echo "Number of AMD/ATI GPUs detected: $gpu_count"

    # Decide how many GPUs to enable based on count.
    if [[ "$gpu_count" -ge 8 ]]; then
        run_tests "0,1,2,3,4,5,6,7"
    elif [[ "$gpu_count" -ge 4 ]]; then
        run_tests "0,1,2,3"
    elif [[ "$gpu_count" -ge 2 ]]; then
        run_tests "0,1"
    else
        run_tests "0"
    fi
}

main "$@"

