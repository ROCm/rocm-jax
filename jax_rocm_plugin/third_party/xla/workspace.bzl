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

# XLA commit info - synced from upstream JAX's MODULE.bazel
# When using bzlmod (default), XLA is loaded via MODULE.bazel
# This file provides commit info for build scripts and WORKSPACE fallback

# buildifier: disable=module-docstring
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Synced from upstream JAX MODULE.bazel
XLA_COMMIT = "ed953c01bb51f95a36abd907d1a64295feef16fc"
XLA_SHA256 = "16699cad982783cdcb68e1e7b98a539a8baa7107428798dbb74b6751041d8fd8"

def repo():
    """Define the XLA repository for WORKSPACE builds (legacy --noenable_bzlmod)."""
    tf_http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-{commit}".format(commit = XLA_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)),
    )
