#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

bazel --bazelrc="${SCRIPT_DIR}/jax.bazelrc" test \
    --config=rocm \
    --@jax//jax:build_jaxlib=wheel \
    --keep_going \
    --test_verbose_timeout_warnings \
    --local_test_jobs=4 \
    --test_output=errors \
    --run_under=@xla//build_tools/rocm:parallel_gpu_execute \
    "$@" \
