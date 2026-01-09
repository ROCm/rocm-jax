#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))
declare -A args
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

WHEELHOUSE=${args['--wheelhouse']}
ROCM_VERSION=${args['--rocm_version']}

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc run \
    --config=rocm \
    "${BAZEL_ARGS[@]}" \
    //pjrt/tools:build_gpu_plugin_wheel \
    -- \
    --output_path="${WHEELHOUSE}" \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=${ROCM_VERSION} \
    --rocm_jax_git_hash=${GIT_HASH}

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc run \
    --config=rocm \
    "${BAZEL_ARGS[@]}" \
    //jaxlib_ext/tools:build_gpu_kernels_wheel \
    -- \
    --output_path="${WHEELHOUSE}" \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=${ROCM_VERSION} \
    --rocm_jax_git_hash=${GIT_HASH}
