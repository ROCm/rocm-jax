#!/usr/bin/env bash

set -ex

PYTHON_VERSION='3.11'
WHEELHOUSE="wheelhouse"

python3 build/build.py build \
    --use_clang=true \
    --clang_path=/lib/llvm-18/bin/clang-18 \
    --python_version=${PYTHON_VERSION} \
    --wheels=jax-rocm-plugin,jax-rocm-pjrt \
    --target_cpu_features=native \
    --rocm_path=/opt/rocm \
    --rocm_version=7 \
    --output_path=${WHEELHOUSE} \
    --verbose

JAX_VERSION='0.8.0'
{
    echo etils
    echo jaxlib=="$JAX_VERSION"
    ls ${WHEELHOUSE}/jax_rocm7_pjrt*${JAX_VERSION}* 2>/dev/null || true
    ls ${WHEELHOUSE}/jax_rocm7_plugin*${JAX_VERSION}* 2>/dev/null || true
} >build/requirements.in

python3 build/build.py requirements_update --python=${PYTHON_VERSION}
python3 build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version=${PYTHON_VERSION} --clang_path=/lib/llvm-18/bin/clang-18

bazel test \
    --config=rocm \
    --test_output=streamed \
    --@jax//jax:build_jaxlib=false \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx902,gfx90a \
    --verbose_failures \
    @jax//tests:ffi_test_gpu
#--override_repository=jax=jax \
