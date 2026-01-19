#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))

TAG_FILTERS="gpu,multiaccelerator,-config-cuda-only"

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc test \
    --config=rocm \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --@jax//jax:build_jaxlib=false \
    --keep_going \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    --local_test_jobs=1 \
    --strategy=TestRunner=local \
    "${BAZEL_ARGS[@]}" \
    -- \
    @jax//tests:gpu_tests
