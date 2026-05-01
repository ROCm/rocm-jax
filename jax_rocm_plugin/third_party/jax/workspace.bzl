load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.10.0 branch)
#   2. Update JAX_COMMIT below

JAX_COMMIT = "b2e32655f8a0643d5132135e3734216d4fd0ac6f"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
    )
