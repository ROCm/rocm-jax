#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))
declare -A args
args['--jax_version']=""
args['--wheelhouse']=""
args['--rocm_version']=""

BAZEL_ARGS=()
while [ $# -gt 0 ]; do
    key=$(echo $1 | cut -d "=" -f 1)
    value=$(echo $1 | cut -d "=" -f 2)
    if [[ -v "args[$key]" ]]; then
        args[$key]=$value
    else
        BAZEL_ARGS+=($1)
    fi
    shift
done

WHEELHOUSE="${args['--wheelhouse']}"
JAX_VERSION=${args['--jax_version']}
ROCM_VERSION=${args['--rocm_version']}

{
    cat build/test-requirements.txt
    ls "${WHEELHOUSE}"/jaxlib-${JAX_VERSION}*.whl
    ls "${WHEELHOUSE}"/jax_rocm${ROCM_VERSION}_pjrt*"${JAX_VERSION}"* 2>/dev/null
    ls "${WHEELHOUSE}"/jax_rocm${ROCM_VERSION}_plugin*"${JAX_VERSION}"* 2>/dev/null
} >build/requirements.in

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc run \
    --config=rocm \
    //build:requirements.update

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc test \
    --config=rocm \
    --@jax//jax:build_jaxlib=false \
    --keep_going \
    --test_verbose_timeout_warnings \
    --local_test_jobs=4 \
    --test_output=errors \
    --run_under=@xla//build_tools/rocm:parallel_gpu_execute \
    "${BAZEL_ARGS[@]}" \
    -- \
    @jax//tests:gpu_tests \
    @jax//tests:backend_independent_tests
