#!/bin/bash

ROCM_JAX_DIR="/rocm-jax"
JAX_DIR="/rocm-jax/jax"
JAX_VERSION="0.7.1"
PYTHON=3.12

pushd "${ROCM_JAX_DIR}" || exit

if [ ! -d "${JAX_DIR}" ]; then
    git clone -b "release/${JAX_VERSION}" --depth 1 "${JAX_DIR}"
fi

pushd "${JAX_DIR}" || exit

# If we haven't stuck the plugin requirements in the requirements.in file yet, put them in
if grep -q jax_rocm7 build/requirements.in; then
    {
        echo "jaxlib == 0.7.1"
        echo "/rocm-jax/wheelhouse/jax_rocm7_pjrt-0.7.1-py3-none-manylinux_2_28_x86_64.whl"
        echo "/rocm-jax/wheelhouse/jax_rocm7_plugin-0.7.1-cp312-cp312-manylinux_2_28_x86_64.whl"
    } >> build/requirements.in
fi

python3 build/build.py requirements_update --python="${PYTHON}"
python3 build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version="${PYTHON}"

./bazel-7.4.1-linux-x86_64 \
    --bazelrc=.bazelrc \
    --bazelrc=../jax_rocm_plugin/rbe.bazelrc \
    test \
    --config=rocm \
    --config=rocm_rbe \
    --//jax:build_jaxlib=false \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx908 \
    --test_output=errors \
    //tests:core_test_gpu
# TODO also run the linalg tests

