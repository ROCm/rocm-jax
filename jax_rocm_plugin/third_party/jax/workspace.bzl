
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "403977d2e252539d2df33cf26d9aeb9dd587e8bf"
SHA = "a8ee2ae32bc3b5f153903326274266e1ea97e345040d360e3631eb5b1a92cf60"

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
        ],
    )
