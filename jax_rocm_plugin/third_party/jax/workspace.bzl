
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "5712de44e97c455faed1fd45532e821ca66d025a"
SHA = "f8fb9e6a7baa789008eb814d138a91340206cbdcebe39c919c0516c8a62a4c65"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
	    "//third_party/jax:0001-Remove-nvidia_wheel_versions.patch",
	    "//third_party/jax:0002-Make-jaxlib-targets-visible.patch",
	    "//third_party/jax:0003-hipblas-typedef-fix.patch",
	    "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
        ],
    )
