
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "21d5e0bf8f07fb7e455f627284e490e6e3068176"
SHA = "a05b6d06ef0609b7a9ad4ea7195dca564e17d99057d589edc9cc8c487d3098de"

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
