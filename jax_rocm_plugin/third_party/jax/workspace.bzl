# Copyright 2024 The JAX Authors.
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

# JAX commit info - synced from upstream jax-ml/jax
# When using bzlmod (default), JAX is loaded via MODULE.bazel
# This file provides commit info for build scripts and WORKSPACE fallback

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Synced from upstream JAX (jax-ml/jax)
COMMIT = "b19be1aa34969e312b7ec30abbae828ec35e1d12"
SHA = "bddf01f474a6105d8a64b6c986fe6220d12291901aeab9c60f6bf15f131118c3"

def repo():
    """Define the JAX repository for WORKSPACE builds (legacy --noenable_bzlmod)."""
    tf_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = tf_mirror_urls("https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)),
    )
