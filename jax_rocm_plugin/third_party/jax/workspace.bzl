load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "98cc66caa859131f1547d885b7817cef993d19a8"
SHA = "c3c640cdac323bbe92b80c697ded4fce0452c884bf08c82445bc8009bb3b7240"

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
