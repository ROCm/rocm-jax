#!/usr/bin/env bash

set -ex

WHEELHOUSE="$(pwd)/wheelhouse"

bazel run \
    --config=rocm \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    --verbose_failures=true \
    --define=xnn_enable_avxvnniint8=false \
    --config=mkl_open_source_only \
    --config=native_arch_posix \
    @jax_rocm_plugin//jaxlib_ext/tools:build_gpu_kernels_wheel \
    -- \
    --output_path="/home/atheodor/projects/rocm-jax/wheelhouse" \
    --cpu=x86_64 \
    --enable-rocm=True \
    --platform_version=7 \
    --rocm_jax_git_hash=333810a7b5b740250aa602a50652d3a261d45cd5

#python3 build/build.py build \
#--use_clang=true \
#--clang_path=/lib/llvm-18/bin/clang-18 \
#--wheels=jax-rocm-plugin,jax-rocm-pjrt \
#--target_cpu_features=native \
#--rocm_path=/opt/rocm \
#--rocm_version=7 \
#--rocm_amdgpu_targets=${AMDGPU_TARGETS} \
#--local_xla_path=${XLA_DIR} \
#--output_path=${WHEELHOUSE} \
#--verbose

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
