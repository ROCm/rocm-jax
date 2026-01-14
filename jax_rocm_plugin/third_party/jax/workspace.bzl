
load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "3f125024ecd17a1d90789a69c59962de4ff6af53"
SHA = "184f45a87463d4e117a515b8d9060602b145f8eed69943add989c7039c1266fe"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/jax-ml/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
	    "//third_party/jax:0001-Remove-nvidia_wheel_versions.patch",
	    "//third_party/jax:0002-Make-jaxlib-targets-visible.patch",
	    #"//third_party/jax:0003-hipblas-typedef-fix.patch",
	    "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
        ],
    )
