#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))
declare -A args
args['--wheelhouse']=""
args['--rocm_version']=""
# we use jax_rocm_plugin hash
# the rest jax, xla are defined as
# a bazel dependency under third_party dir
args['--git_hash']=""

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
GIT_HASH=${args['--git_hash']}

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc run \
    --config=rocm \
    "${BAZEL_ARGS[@]}" \
    @jax//jaxlib/tools:build_wheel_tool \
    -- \
    --jaxlib_git_hash="this lib is built as a part of jax_rocm_plugin build using commit [${GIT_HASH}], please check third_party jax dependency" \
    --output_path="${WHEELHOUSE}" \
    --cpu=x86_64

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
