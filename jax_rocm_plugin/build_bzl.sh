#!/usr/bin/env bash

set -ex

# Configuration
PYTHON_VERSION='3.12'
ROCM_VERSION='7'
JAX_VERSION='0.8.0'

WHEELHOUSE=$(mktemp -d)

# Get git hash for wheel metadata
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# Cleanup function
clean_up() {
    rm -rf "${WHEELHOUSE}"
    rm -f build/requirements.in
}

trap clean_up EXIT

# Create wheelhouse directory
mkdir -p "${WHEELHOUSE}"

bazel run \
    --config=rocm \
    --config=rocm_rbe \
    //pjrt/tools:build_gpu_plugin_wheel \
    -- \
    --output_path="${WHEELHOUSE}" \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=${ROCM_VERSION} \
    --rocm_jax_git_hash=${GIT_HASH}

bazel run \
    --config=rocm \
    --config=rocm_rbe \
    //jaxlib_ext/tools:build_gpu_kernels_wheel \
    -- \
    --output_path="${WHEELHOUSE}" \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=${ROCM_VERSION} \
    --rocm_jax_git_hash=${GIT_HASH}

{
    cat build/test-requirements.txt
    echo opt-einsum
    echo etils
    echo zstandard
    echo "jaxlib==${JAX_VERSION}"
    ls "${WHEELHOUSE}"/jax_rocm${ROCM_VERSION}_pjrt*"${JAX_VERSION}"* 2>/dev/null || true
    ls "${WHEELHOUSE}"/jax_rocm${ROCM_VERSION}_plugin*"${JAX_VERSION}"* 2>/dev/null || true
} >build/requirements.in

bazel run \
    --config=rocm \
    --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION} \
    //build:requirements.update

bazel test \
    --config=rocm \
    --config=rocm_rbe \
    --test_output=streamed \
    --@jax//jax:build_jaxlib=false \
    --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION} \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx902,gfx90a \
    --verbose_failures \
    --strategy=TestRunner=local @jax//tests:gpu_tests
