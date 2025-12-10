
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "00e65c328ec9a82428a9f37e38c8583ec606038d"
SHA = "e95ff2282014bb6b3246fcd633340a5a20d72c2c974e8e10b81277462a1d1bcf"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
           "//third_party/jax:build.patch",
           "//third_party/jax:jax_bzl.patch",
           "//third_party/jax:hipBlas_typedef.patch",
           "//third_party/jax:no_gpu_fail.patch"
        ],
    )
