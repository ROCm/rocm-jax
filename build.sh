#!/usr/bin/env bash

set -ex

WHEELHOUSE="$(pwd)/wheelhouse"
pushd jax_rocm_plugin

python3 build/build.py build \
    --use_clang=true \
    --clang_path=/lib/llvm-18/bin/clang-18 \
    --wheels=jax-rocm-plugin,jax-rocm-pjrt \
    --target_cpu_features=native \
    --rocm_path=/opt/rocm \
    --rocm_version=7 \
    --output_path=${WHEELHOUSE} \
    --verbose
popd

JAX_VERSION='7'
{
    echo etils
    echo jaxlib=="$JAX_VERSION"
    ls ${WHEELHOUSE}/jax_rocm7_pjrt*${JAX_VERSION}*
    ls ${WHEELHOUSE}/jax_rocm7_plugin*$JAX_VERSION*${PYTHON//./}*
} >build/requirements.in

python3 build/build.py requirements_update --python="3.11"
python3 build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version="3.11"

bazel test \
    --config=rocm \
    --test_output=streamed \
    --@jax//jax:build_jaxlib=false \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx902,gfx90a \
    --verbose_failures \
    @jax//tests:ffi_test_gpu
#--override_repository=jax=jax \
