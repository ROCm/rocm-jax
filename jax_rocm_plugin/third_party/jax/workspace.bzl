load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.9.2 branch)
#   2. Update JAX_COMMIT below

JAX_COMMIT = "dbc860fefaaa8a18ebf1f0bedc3cfe5ca3e17558"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
        patch_tool = "patch",
        patch_args = ["-p1"],
        patches = [
            "//third_party/jax:0008-Expose-rocm-plugin-targets.patch",
        ],
    )
