#!/usr/bin/env bash

TARGETS_TO_IGNORE=(
    -@jax//tests:export_harnesses_multi_platform_test_gpu
    -@jax//tests:gpu_memory_flags_test_gpu
    -@jax//tests:lax_numpy_test_gpu
    -@jax//tests:lax_test_gpu
    -@jax//tests:linalg_test_gpu
    -@jax//tests:aot_test_gpu
    -@jax//tests:buffer_callback_test_gpu
    -@jax//tests:export_harnesses_multi_platform_test_gpu
    -@jax//tests:lax_control_flow_test_gpu
    -@jax//tests:export_test_gpu
    -@jax//tests:gpu_memory_flags_test_gpu
    -@jax//tests:lax_numpy_test_gpu
    -@jax//tests:lax_test_gpu
    -@jax//tests:lobpcg_test_gpu
    -@jax//tests:ode_test_gpu
    -@jax//tests:lax_scipy_special_functions_test_gpu
    -@jax//tests:svd_test_gpu
)

echo "${TARGETS_TO_IGNORE[*]}"
