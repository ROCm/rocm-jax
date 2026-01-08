#!/usr/bin/env bash

set -ex

WHEELHOUSE="$(pwd)/wheelhouse"

bazel run \
    --config=rocm_wheels \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    @jax_rocm_plugin//jaxlib_ext/tools:build_gpu_kernels_wheel \
    -- \
    --output_path=${WHEELHOUSE} \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=7 \
    --rocm_jax_git_hash=

bazel run \
    --config=rocm_wheels \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    @jax_rocm_plugin//pjrt/tools:build_gpu_plugin_wheel \
    -- \
    --output_path=${WHEELHOUSE} \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=7 \
    --rocm_jax_git_hash=

JAX_VERSION=${args['--jax_version']}
{
    echo etils
    echo jaxlib=="$JAX_VERSION"
    ls ${WHEELHOUSE}/jax_rocm7_pjrt*${JAX_VERSION}*
    ls ${WHEELHOUSE}/jax_rocm7_plugin*$JAX_VERSION*${PYTHON//./}*
} >build/requirements.in

python3 build/build.py requirements_update --python="3.11"
python3 build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version="3.11"
popd

bazel test \
    --config=rocm \
    --test_output=streamed \
    --@jax//jax:build_jaxlib=false \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx902,gfx90a \
    --verbose_failures \
    @jax//tests:ffi_test_gpu
#--override_repository=jax=jax \
