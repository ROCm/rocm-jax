load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "cef02f3ae4abf69862294cca9370c721eb7eb2b4"
SHA = "c3b9d0172cefede0626f2bb180b6c80e2d6cf7f0a5f71eb69a6cff23a358ed91"

def repo():
    amd_http_archive(
        name = "jax",
        sha256 = SHA,
        strip_prefix = "jax-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/ROCm/jax/archive/{commit}.tar.gz".format(commit = COMMIT)],
        patch_file = [
	        "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
            "//third_party/jax:0006-Enable-testing-with-ROCm-plugin-wheels.patch",
        ],
    )
