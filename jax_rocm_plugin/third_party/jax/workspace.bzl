load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.9.2 branch)
#   2. Update JAX_COMMIT below

JAX_COMMIT = "53ae1b7b02a411c37d9e7ff1539b2ecfc33337ad"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
    )
