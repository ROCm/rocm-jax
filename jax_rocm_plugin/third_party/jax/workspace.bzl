load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.9.2 branch)
#   2. Update JAX_COMMIT below

JAX_COMMIT = "aae3e281b52bcab00661666255e80b4fa9751b70"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
    )
