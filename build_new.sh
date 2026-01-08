#!/usr/bin/env bash

set -ex

WHEELHOUSE="$(pwd)/wheelhouse"

bazel run \
    --config=rocm_wheels \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    @jax_rocm_plugin//jaxlib_ext/tools:build_gpu_kernels_wheel \
    -- \
    --output_path=${WHEELHOUSE} \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=7 \
    --rocm_jax_git_hash=

bazel run \
    --config=rocm_wheels \
    --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
    @jax_rocm_plugin//pjrt/tools:build_gpu_plugin_wheel \
    -- \
    --output_path=${WHEELHOUSE} \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=7 \
    --rocm_jax_git_hash=

JAX_VERSION=0.8.0
PYTHON_VERSION=3.11
PYTHON_TAG=cp311
PROJECT_ROOT="$(pwd)"
{
    echo etils
    echo jaxlib=="$JAX_VERSION"
    # Use file:/// URIs (three slashes) for absolute paths to avoid pip-tools URI issues
    ls ${WHEELHOUSE}/jax_rocm7_pjrt*${JAX_VERSION}* 2>/dev/null || true
    ls ${WHEELHOUSE}/jax_rocm7_plugin*${JAX_VERSION}*${PYTHON_TAG}* 2>/dev/null || true
} >build/requirements.in

bazel run --repo_env=HERMETIC_PYTHON_VERSION=3.11 --verbose_failures=true //build:requirements.update
# Ensure bazel configuration is set up (equivalent to build.py --configure_only)
bazel query --repo_env=HERMETIC_PYTHON_VERSION=3.11 --config=rocm_wheels @jax_rocm_plugin//jaxlib_ext/tools:build_gpu_kernels_wheel >/dev/null

bazel test \
    --config=rocm \
    --test_output=streamed \
    --@jax//jax:build_jaxlib=false \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx902,gfx90a \
    --verbose_failures \
    @jax//tests:ffi_test_gpu
#--override_repository=jax=jax \
