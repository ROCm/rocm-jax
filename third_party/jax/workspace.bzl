load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def jax_workspace():
    new_git_repository(
        name = "jax",
        commit = "ad54f8b18bf44ad8511b59483baa73e6d2318093",
        remote = "https://github.com/ROCm/jax.git",
        patches = [
            "//third_party/jax:0001-Enable-testing-with-ROCm-plugin-wheels.patch",
        ],
    )
