
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "6821ce3e65986a9146b6864cc0540a7d6abaa76e"
SHA = "f468fecd52065b44274a5f604cc99aad0c32138610d44a5b815c695430760373"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/ROCm/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
	    "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
        ],
    )
