load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update JAX:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.9.1 branch)
#   2. Update JAX_COMMIT below

JAX_COMMIT = "a5a5bfd0c1cef100b4f775bf6a2b3e9aa8327ae1"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
        patch_tool = "patch",
        patch_args = ["-p1"],
        patches = [
            "//third_party/jax:0005-Fix-HIP-availability-errors.patch", #TODO(gulsumgudukbay): check if this is still needed
            "//third_party/jax:0008-Expose-rocm-plugin-targets.patch",
        ],
    )
