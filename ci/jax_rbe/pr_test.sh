#!/bin/bash

set -ex

args=("$@")
JAX_VERSION="${args[0]}"
PYTHON="${args[1]}"

ROCM_JAX_DIR=$(realpath ".")
JAX_DIR=$(realpath "./jax")
WHEELHOUSE=$(realpath "./wheelhouse")


pushd "${ROCM_JAX_DIR}" || exit

if [ ! -d "${JAX_DIR}" ]; then
    git clone -b "release/${JAX_VERSION}" --depth 1 "${JAX_DIR}"
fi

pushd "${JAX_DIR}" || exit

# If we haven't stuck the plugin requirements in the requirements.in file yet, put them in.
# Disable shellcheck's quoted string rule. Strings in the {} block need to be unquoted in order
# for the * to expand.
# shellcheck disable=SC2086
if ! grep -q jax_rocm7 build/requirements.in; then
    {
        echo "jaxlib==${JAX_VERSION}"
	echo ${WHEELHOUSE}/jax_rocm7_pjrt*${JAX_VERSION}*
	echo ${WHEELHOUSE}/jax_rocm7_plugin*${JAX_VERSION}*${PYTHON//.}*
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
    --noremote_accept_cached \
    --//jax:build_jaxlib=false \
    --action_env=TF_ROCM_AMDGPU_TARGETS="gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201" \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    //tests:core_test_gpu \
    //tests:linalg_test_gpu \
    --test_filter=CoreTest \
    --test_filter=JaxprTypeChecks \
    --test_filter=DynamicShapesTest \
    --test_filter=testMatmul \
    //tests:ffi_test_gpu

