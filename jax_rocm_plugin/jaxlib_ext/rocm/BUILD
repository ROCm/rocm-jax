# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Local nanobind_extension wrappers for JAX ROCm GPU kernels.
#
# These targets mirror @jax//jaxlib/rocm but allow us to control
# features and linkopts via select() on rocm_path_type:
#   - link_only  → strip solib rpaths, embed wheel-specific rpaths
#   - (default)  → preserve original Bazel rpath behavior

load("@rules_python//python:defs.bzl", "py_library")
load("//build:rpath.bzl", "rocm_nanobind_extension")

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

rocm_nanobind_extension(
    name = "_rnn",
    srcs = ["@jax//jaxlib/gpu:rnn.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_rnn",
    deps = [
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib/rocm:miopen_rnn_kernels",
        "@jax//jaxlib:absl_status_casters",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@nanobind",
    ],
)

rocm_nanobind_extension(
    name = "_solver",
    srcs = ["@jax//jaxlib/gpu:solver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_solver",
    deps = [
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib/rocm:hipsolver_kernels_ffi",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@local_config_rocm//rocm:hipblas",
        "@local_config_rocm//rocm:hipsolver",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
    ],
)

rocm_nanobind_extension(
    name = "_sparse",
    srcs = ["@jax//jaxlib/gpu:sparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_sparse",
    deps = [
        "@jax//jaxlib/rocm:hip_gpu_kernel_helpers",
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib/rocm:hipsparse_kernels",
        "@jax//jaxlib:absl_status_casters",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_rocm//rocm:hipsparse",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

rocm_nanobind_extension(
    name = "_linalg",
    srcs = ["@jax//jaxlib/gpu:linalg.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_linalg",
    deps = [
        "@jax//jaxlib/rocm:hip_gpu_kernel_helpers",
        "@jax//jaxlib/rocm:hip_linalg_kernels",
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
    ],
)

rocm_nanobind_extension(
    name = "_prng",
    srcs = ["@jax//jaxlib/gpu:prng.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_prng",
    deps = [
        "@jax//jaxlib/rocm:hip_gpu_kernel_helpers",
        "@jax//jaxlib/rocm:hip_prng_kernels",
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:hip_runtime",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
    ],
)

rocm_nanobind_extension(
    name = "_hybrid",
    srcs = ["@jax//jaxlib/gpu:hybrid.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = [
        "-L/opt/rocm/lib",
        "-lamdhip64",
    ],
    module_name = "_hybrid",
    deps = [
        "@jax//jaxlib/rocm:hip_gpu_kernel_helpers",
        "@jax//jaxlib/rocm:hip_hybrid_kernels",
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@jax//jaxlib/cpu:lapack_kernels",
        "@com_google_absl//absl/base",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:hip_runtime",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
        "@xla//xla/ffi/api:ffi",
    ],
)

rocm_nanobind_extension(
    name = "_triton",
    srcs = ["@jax//jaxlib/gpu:triton.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_triton",
    deps = [
        "@jax//jaxlib/rocm:hip_gpu_kernel_helpers",
        "@jax//jaxlib/rocm:hip_vendor",
        "@jax//jaxlib/rocm:triton_kernels",
        "@jax//jaxlib/rocm:triton_utils",
        "@jax//jaxlib:absl_status_casters",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@jax//jaxlib/gpu:triton_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:hip_runtime",
        "@nanobind",
    ],
)

rocm_nanobind_extension(
    name = "rocm_plugin_extension",
    srcs = ["rocm_plugin_extension.cc"],
    module_name = "rocm_plugin_extension",
    deps = [
        "@jax//jaxlib/rocm:hip_gpu_kernel_helpers",
        "@jax//jaxlib/rocm:py_client_gpu",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@jax//jaxlib/gpu:gpu_plugin_extension",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:hip_runtime",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
    ],
)

py_library(
    name = "rocm_gpu_support",
    deps = [
        ":_hybrid",
        ":_linalg",
        ":_prng",
        ":_rnn",
        ":_solver",
        ":_sparse",
        ":_triton",
    ],
)
