
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "127aa7621868cb77e552b5d1f90e4a42b09c13fa"
SHA = "80840d370d22814a5a895331cad081966c08e3e6468290b2f2f836a8c9f83e83"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
           "//third_party/jax:build.patch",
        ],
    )
