#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))

TAG_FILTERS="gpu,-multiaccelerator,-config-cuda-only"

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc test \
    --config=rocm \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --@jax//jax:build_jaxlib=built_in_wheels \
    --keep_going \
    --test_verbose_timeout_warnings \
    --local_test_jobs=4 \
    --test_output=errors \
    --run_under=@xla//build_tools/rocm:parallel_gpu_execute \
    "${BAZEL_ARGS[@]}" \
    -- \
    @jax//tests:gpu_tests \
    @jax//tests:backend_independent_tests
