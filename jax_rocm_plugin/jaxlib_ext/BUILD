# Copyright 2018 The JAX Authors.
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

# AMD HIP kernels

load("@jax//jaxlib:jax.bzl", "nanobind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

nanobind_extension(
    name = "rocm_plugin_extension",
    srcs = ["rocm_plugin_extension.cc"],
    module_name = "rocm_plugin_extension",
    deps = [
        ":py_client_gpu",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@jax//jaxlib:kernel_nanobind_helpers",
        "@jax//jaxlib/gpu:gpu_plugin_extension",
        "@local_config_rocm//rocm:hip_runtime",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
    ],
)
