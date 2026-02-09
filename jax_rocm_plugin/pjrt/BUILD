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

load("//build:rpath.bzl", "patched_rpath_binary")

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "pjrt_c_api_gpu_plugin_raw.so",
    linkopts = [
        "-Wl,--version-script,$(location :gpu_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    deps = [
        ":gpu_version_script.lds",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_version_script.lds",
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/stream_executor:rocm_platform",
    ],
)

patched_rpath_binary(
    name = "pjrt_c_api_gpu_plugin.so",
    src = ":pjrt_c_api_gpu_plugin_raw.so",
)
