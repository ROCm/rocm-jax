
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
	    "//third_party/jax:0001-Remove-nvidia_wheel_versions.patch",
	    "//third_party/jax:0002-Make-jaxlib-targets-visible.patch",
	    "//third_party/jax:0003-hipblas-typedef-fix.patch",
	    "//third_party/jax:0004-No-GPU-fail.patch",
	    "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
        ],
    )
