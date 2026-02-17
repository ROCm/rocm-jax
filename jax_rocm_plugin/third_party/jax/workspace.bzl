load("//third_party:repo.bzl", "amd_http_archive")

COMMIT = "fbfa695aea59ed578b81d8fc72ab23bba5d2cfaa"
SHA = "b740b326b468ce7ef967fbfab0accfb19850fab9c43ab6a3a37112eff34223c2"

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/ROCm/jax.git",
        commit = JAX_COMMIT,
        patches = [
            "//third_party/jax:0005-Fix-HIP-availability-errors.patch",
            "//third_party/jax:0006-Enable-testing-with-ROCm-plugin-wheels.patch",  # TODO: remove due to: https://github.com/jax-ml/jax/pull/34641
            "//third_party/jax:0007-Fix-legacy-create-init.patch",  # TODO: remove due to: https://github.com/jax-ml/jax/pull/34770
        ],
    )
