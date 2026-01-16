
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "9b33362373387dc915b1e472a5e7392f70cc2b7d"
SHA = "0a723a725b60de0702be442cac270b074c39d26f2bc7a023b79dedc24a59b163"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
	    "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
        ],
    )
