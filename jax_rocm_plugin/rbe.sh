#!/bin/bash

./bazel-7.4.1-linux-x86_64 run --repo_env=HERMETIC_PYTHON_VERSION=3.10 --verbose_failures=true --action_env=CLANG_COMPILER_PATH="/usr/lib/llvm-18/bin/clang" --repo_env=CC="/usr/lib/llvm-18/bin/clang" --repo_env=BAZEL_COMPILER="/usr/lib/llvm-18/bin/clang" --config=clang --define=xnn_enable_avxvnniint8=false --config=mkl_open_source_only --config=avx_posix --config=rocm_base --config=rocm --action_env=CLANG_COMPILER_PATH="/usr/lib/llvm-18/bin/clang" \
	--config=rbe_linux_x86_64 \
	--sandbox_debug \
	--override_repository=xla=../xla \
	--repo_env=HIPCC=/opt/rocm/bin/hipcc \
	--config=rocm_clang_official_hermetic \
	--action_env=ROCM_PATH="/opt/rocm/" --action_env=TF_ROCM_AMDGPU_TARGETS=gfx90a //pjrt/tools:build_gpu_plugin_wheel -- --output_path="/home/chahofer/dockerx/rocm/rocm-jax/jax_rocm_plugin/dist" --cpu=x86_64 --enable-rocm=True --platform_version=7 --jaxlib_git_hash=cb024e300c299b8968d7f5788533e1784c468104

