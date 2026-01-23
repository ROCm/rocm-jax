#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc test \
    --config=rocm \
    --repo_env="PLUGIN_WHEEL_DEPS=@jax_rocm_plugin//:pjrt.whl,@jax_rocm_plugin//:plugin.whl,@jax_rocm_plugin//:jaxlib.whl" \
    --@jax//jax:build_jaxlib=plugin_wheels \
    --keep_going \
    --test_verbose_timeout_warnings \
    --local_test_jobs=4 \
    --test_output=errors \
    --run_under=@xla//build_tools/rocm:parallel_gpu_execute \
    "$@" \
    -- \
    @jax//tests:gpu_tests \
    @jax//tests:backend_independent_tests
