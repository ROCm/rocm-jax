load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.8.2 branch)
#   2. Update JAX_COMMIT below

JAX_COMMIT = "32ba3e9524b0089eed8fe309468de523e4157023"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
        patch_tool = "patch",
        patch_args = ["-p1"],
        patches = [
            "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
            "//third_party/jax:0006-Enable-testing-with-ROCm-plugin-wheels.patch",  # TODO: remove due to: https://github.com/jax-ml/jax/pull/34641
        ],
    )
