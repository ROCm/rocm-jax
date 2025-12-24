workspace(name = "rocm_jax")

# -----------------------------------------------------------------------------
# Hermetic Python (FIRST)
# -----------------------------------------------------------------------------

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

new_git_repository(
    name = "jax",
    commit = "ad54f8b18bf44ad8511b59483baa73e6d2318093",
    remote = "https://github.com/ROCm/jax.git",
)

new_git_repository(
    name = "xla",
    commit = "96b4835e5e31e878132524104d8cbf364f6d4662",
    remote = "https://github.com/ROCm/xla.git",
)

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

local_repository(
    name = "local_jax_rocm_plugin",
    path = "jax_rocm_plugin",
)

# You may still need bazel_skylib and rules_cc for Python toolchains
http_archive(
    name = "bazel_skylib",
    strip_prefix = "bazel-skylib-1.5.0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/refs/tags/1.5.0.tar.gz"],
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-0.0.17",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.0.17.tar.gz"],
)

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "../dist",
    local_wheel_inclusion_list = [
        "ml_dtypes*",
        "numpy*",
        "scipy*",
    ],
    local_wheel_workspaces = [],
    requirements = {
        "3.11": "@jax//:build/requirements_lock_3_11.txt",
        "3.12": "@jax//:build/requirements_lock_3_12.txt",
        "3.13": "@jax//:build/requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# -----------------------------------------------------------------------------
# Core infrastructure
# -----------------------------------------------------------------------------

# rules_ml_toolchain, etc.
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "3ea1041deb46cf4f927dd994e32acd8c436f7997b12a9558e85dee6a5a89e35c",
    strip_prefix = "rules_ml_toolchain-0fccc2447ef3bec3d75046a60a1895f053424727",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/0fccc2447ef3bec3d75046a60a1895f053424727.tar.gz",
    ],
)

# -----------------------------------------------------------------------------
# C++ toolchains
# -----------------------------------------------------------------------------

load("@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl", "cc_toolchain_deps")

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

# -----------------------------------------------------------------------------
# Flatbuffers
# -----------------------------------------------------------------------------

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")

flatbuffers()

# -----------------------------------------------------------------------------
# JAX wheel glue
# -----------------------------------------------------------------------------

load("@jax//jaxlib:jax_python_wheel.bzl", "jax_python_wheel_repository")

jax_python_wheel_repository(
    name = "jax_wheel",
    version_key = "_version",
    version_source = "@jax//jax:version.py",
)

load("@xla//third_party/py:python_wheel.bzl", "python_wheel_version_suffix_repository")

python_wheel_version_suffix_repository(
    name = "jax_wheel_version_suffix",
)

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@xla//third_party/nvshmem/hermetic:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@xla//third_party/nvshmem/hermetic:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

load("@jax//:test_shard_count.bzl", "test_shard_count_repository")

test_shard_count_repository(
    name = "test_shard_count",
)

#load(
#"@xla//third_party/nvshmem/hermetic:nvshmem_configure.bzl",
#"nvshmem_configure",
#)

#nvshmem_configure(name = "local_config_nvshmem")
