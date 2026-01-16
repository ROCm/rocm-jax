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
    ls "${WHEELHOUSE}"/jax_rocm${ROCM_VERSION}_pjrt*"${JAX_VERSION}"*.whl
    ls "${WHEELHOUSE}"/jax_rocm${ROCM_VERSION}_plugin*"${JAX_VERSION}"*.whl
} >build/requirements.in

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc run \
    --config=rocm \
    "${BAZEL_ARGS[@]}" \
    //build:requirements.update

TAG_FILTERS="gpu,multiaccelerator,-config-cuda-only"

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc test \
    --config=rocm \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --@jax//jax:build_jaxlib=false \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    --local_test_jobs=1 \
    --strategy=TestRunner=local \
    "${BAZEL_ARGS[@]}" \
    -- \
    @jax//tests:gpu_tests
