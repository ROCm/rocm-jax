
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "8bc0ae6da4f6d334d03be7b76be841d72a56d330"
SHA = "0face25c515fdb2762fdd223192eaef9ace2aa0bc2009a2beaaf92fac5ad135b"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
           "//third_party/jax:build.patch",
           "//third_party/jax:hipBlas_typedef.patch"
        ],
    )
