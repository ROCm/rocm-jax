#!/bin/bash

set -ex
shopt -s nullglob

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

# Pin jaxlib from PyPI here. The rocm plugin/pjrt wheels are NOT appended to
# requirements.in: they're injected into the merged requirements as proper
# `==<version>` entries by xla's _get_injected_local_wheels via the
# jax/dist symlink set up in pr_setup.sh. This is what creates the
# @pypi_jax_rocm7_{plugin,pjrt} repos that //jaxlib/tools:rocm_plugin_*_wheel
# expects (the `@ file://` form pip-compile would otherwise produce does not
# get a per-package repo in rules_python 1.8.5).
if ! grep -q jax_rocm7 build/requirements.in; then
    # Sanity-check the rocm wheels are present where pr_setup.sh expects them.
    pjrt=( "${WHEELHOUSE}"/jax_rocm7_pjrt-*"${JAX_VERSION}"*manylinux_2_28*.whl )
    plugin=( "${WHEELHOUSE}"/jax_rocm7_plugin-*"${JAX_VERSION}"*cp"${PYTHON//.}"*manylinux_2_28*.whl )

    (( ${#pjrt[@]} == 1 )) || { echo "Expected 1 pjrt wheel, found ${#pjrt[@]}: ${pjrt[*]}"; exit 1; }
    (( ${#plugin[@]} == 1 )) || { echo "Expected 1 plugin wheel, found ${#plugin[@]}: ${plugin[*]}"; exit 1; }

    {
        echo "jaxlib==${JAX_VERSION}"
        echo "# jax_rocm7_{plugin,pjrt} injected via jax/dist symlink (see pr_setup.sh)"
    } >> build/requirements.in
fi

python3 build/build.py requirements_update --python="${PYTHON}"
python3 build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version="${PYTHON}"

bazel \
    --bazelrc=../jax_rocm_plugin/rbe.bazelrc \
    test \
    --config=rocm \
    --config=rocm_rbe \
    --noremote_accept_cached \
    --//jax:build_jaxlib=false \
    --action_env=TF_ROCM_AMDGPU_TARGETS="gfx908,gfx90a,gfx9-4-generic,gfx10-3-generic,gfx11-generic,gfx12-generic" \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    //tests:core_test_gpu \
    //tests:linalg_test_gpu \
    --test_filter=CoreTest \
    --test_filter=JaxprTypeChecks \
    --test_filter=DynamicShapesTest \
    --test_filter=testMatmul \
    //tests:ffi_test_gpu
