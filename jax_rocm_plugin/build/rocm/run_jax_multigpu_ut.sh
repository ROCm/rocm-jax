#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath $(dirname $0))
declare -A args
args['--python_version']=""
args['--jax_version']=""
args['--jax_dir']=""
args['--xla_dir']=""

while [ $# -gt 0 ]; do
    key=$(echo $1 | cut -d "=" -f 1)
    value=$(echo $1 | cut -d "=" -f 2)
    if [[ -v "args[$key]" ]]; then
        args[$key]=$value
    else
        echo "Unexpected argument $1"
        exit 1
    fi
    shift
done

SCRIPT_DIR=$(realpath $(dirname $0))
JAX_DIR=${args['--jax_dir']}
WHEELHOUSE="${JAX_DIR}/../wheelhouse"
XLA_DIR=${args['--xla_dir']}

clean_up() {
    rm -rf ${WHEELHOUSE}
}

pushd ${JAX_DIR}
JAX_VERSION=${args['--jax_version']}
{
    echo etils
    echo jaxlib=="$JAX_VERSION"
    ls ${WHEELHOUSE}/jax_rocm7_pjrt*${JAX_VERSION}*
    ls ${WHEELHOUSE}/jax_rocm7_plugin*$JAX_VERSION*${PYTHON//./}*
} >build/requirements.in

python3 build/build.py requirements_update --python="${PYTHON}"
python3 build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version="${PYTHON}" --local_xla_path=${XLA_DIR}

bazel --bazelrc=${SCRIPT_DIR}/jax.bazelrc test \
    --config=rocm \
    --config=rocm_rbe \
    --spawn_strategy=local \
    --strategy=TestRunner=local \
    --//jax:build_jaxlib=false \
    --override_repository=xla=${XLA_DIR} \
    --build_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --test_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --action_env=TF_ROCM_AMDGPU_TARGETS="gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201" \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    -- \
    //tests:multiprocess_gpu_test \
    //tests:debug_info_test_gpu \
    //tests:random_test_gpu \
    //tests:jax_jit_test_gpu \
    //tests:mesh_utils_test \
    //tests:pjit_test_gpu \
    //tests:linalg_sharding_test_gpu \
    //tests:multi_device_test_gpu \
    //tests:distributed_test_gpu \
    //tests:shard_alike_test_gpu \
    //tests:api_test_gpu \
    //tests:ragged_collective_test_gpu \
    //tests:batching_test_gpu \
    //tests:scaled_matmul_stablehlo_test_gpu \
    //tests:export_harnesses_multi_platform_test_gpu \
    //tests:pickle_test_gpu \
    //tests:roofline_test_gpu \
    //tests:profiler_test_gpu \
    //tests:error_check_test_gpu \
    //tests:debug_nans_test_gpu \
    //tests:shard_map_test_gpu \
    //tests:cudnn_fusion_test_gpu \
    //tests:compilation_cache_test_gpu \
    //tests:export_back_compat_test_gpu \
    //tests:pgle_test_gpu \
    //tests:ffi_test_gpu \
    //tests:fused_attention_stablehlo_test_gpu \
    //tests:layout_test_gpu \
    //tests:pmap_test_gpu \
    //tests:aot_test_gpu \
    //tests:mock_gpu_topology_test_gpu \
    //tests:ann_test_gpu \
    //tests:debugging_primitives_test_gpu \
    //tests:array_test_gpu \
    //tests:export_test_gpu \
    //tests:memories_test_gpu \
    //tests:debugger_test_gpu \
    //tests:lax_control_flow_test_gpu \
    //tests:checkify_test_gpu
#//tests:colocated_gputhon_test
#//tests_gputhon_callback_test_gpu
#//tests/mosaic:gpu_test_gpu \
