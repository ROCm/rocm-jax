# Copyright 2025 The JAX Authors.
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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

XLA_BRANCH = "rocm-jaxlib-v0.7.0"

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/ROCm/xla.git",
        branch = "{xla_branch}".format(xla_branch = XLA_BRANCH)
        # init_submodules = True,      # if the repo uses submodules
        # Optional: apply patches to the checkout
        # patches = ["@//third_party:xla_fix.patch"],
        # patch_args = ["-p1"],
    )