#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

TAG_FILTERS=jax_test_gpu,-config-cuda-only,-manual

TARGETS_TO_IGNORE=()

for arg in "$@"; do
    if [[ "$arg" == "--config" ]]; then
        echo "Invalid config format, configs must be in a form --config=value"
        exit 255
    fi
    if [[ "$arg" == "--config=rocm_mgpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multiaccelerator"
    fi
    if [[ "$arg" == "--config=rocm_sgpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-multiaccelerator"
    fi
    if [[ "$arg" == "--config=asan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-noasan"
        TARGETS_TO_IGNORE+=(
            -@jax//tests:memories_test_gpu
            -@jax//tests:python_callback_test_gpu
            -@jax//tests:shard_alike_test_gpu
            -@jax//tests:linalg_sharding_test_gpu
            -@jax//tests:pmap_without_shmap_test_gpu
            -@jax//tests:array_test_gpu
            -@jax//tests:custom_partitioning_test_gpu
            -@jax//tests:pmap_test_gpu
            -@jax//tests:ragged_collective_test_gpu
            -@jax//tests:layout_test_gpu
            -@jax//tests:profiler_test_gpu
            -@jax//tests:ragged_collective_test_gpu
            -@jax//tests:pmap_test_gpu
            -@jax//tests:custom_partitioning_test_gpu
            -@jax//tests:array_test_gpu
            -@jax//tests:pmap_without_shmap_test_gpu
            -@jax//tests:memories_test_gpu
            -@jax//tests:pjit_test_gpu
            -@jax//tests:debugging_primitives_test_gpu
            -@jax//tests:shard_map_test_gpu
        )
    fi
done

bazel --bazelrc="${SCRIPT_DIR}/jax.bazelrc" test \
    --config=rocm \
    --test_tag_filters="${TAG_FILTERS}" \
    --build_tag_filters="${TAG_FILTERS}" \
    --@jax//jax:build_jaxlib=wheel \
    --keep_going \
    --test_verbose_timeout_warnings \
    --local_test_jobs=4 \
    --test_timeout=920,2400,7200,9600 \
    --test_sharding_strategy=disabled \
    --flaky_test_attempts=3 \
    --test_output=errors \
    --run_under=@xla//build_tools/rocm:parallel_gpu_execute \
    "$@" \
    "${TARGETS_TO_IGNORE[@]}"
