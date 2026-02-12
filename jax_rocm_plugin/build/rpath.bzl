load("@jax//jaxlib:jax.bzl", "nanobind_extension")

_ROCM_LINK_ONLY = "@local_config_rocm//rocm:link_only"

_WHEEL_RPATHS = [
    "-Wl,-rpath,$$ORIGIN/../rocm/lib",
    "-Wl,-rpath,$$ORIGIN/../../rocm/lib",
    "-Wl,-rpath,/opt/rocm/lib",
]

def _wheel_features():
    return select({
        _ROCM_LINK_ONLY: ["no_solib_rpaths"],
        "//conditions:default": [],
    })

def _wheel_linkopts():
    return select({
        _ROCM_LINK_ONLY: _WHEEL_RPATHS,
        "//conditions:default": [],
    })

def rocm_cc_binary(name, features = [], linkopts = [], **kwargs):
    """cc_binary that automatically strips solib rpaths and embeds wheel RPATHs.

    Args:
        name: Target name.
        features: Additional features (rpath features are appended automatically).
        linkopts: Additional linkopts (wheel RPATHs are appended automatically).
        **kwargs: Passed through to native.cc_binary.
    """
    native.cc_binary(
        name = name,
        features = features + _wheel_features(),
        linkopts = linkopts + _wheel_linkopts(),
        **kwargs
    )

def rocm_nanobind_extension(name, features = [], linkopts = [], **kwargs):
    """nanobind_extension that automatically strips solib rpaths and embeds wheel RPATHs.

    Args:
        name: Target name.
        features: Additional features (rpath features are appended automatically).
        linkopts: Additional linkopts (wheel RPATHs are appended automatically).
        **kwargs: Passed through to nanobind_extension.
    """
    nanobind_extension(
        name = name,
        features = features + _wheel_features(),
        linkopts = linkopts + _wheel_linkopts(),
        **kwargs
    )
