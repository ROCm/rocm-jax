load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.8.2 branch)
#   2. Update JAX_COMMIT below
#
# No SHA256 computation needed when using git_repository.

JAX_COMMIT = "2890b29ceba52e4006144d658bf9cd777c7f3867"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
    )
