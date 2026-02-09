"""Macro to patch RPATH of shared libraries using patchelf."""

_ROCM_RPATH = "$$ORIGIN/../rocm/lib:$$ORIGIN/../../rocm/lib:/opt/rocm/lib"

def patched_rpath_binary(name, src, visibility = None):
    """Copies a shared library and patches its RPATH using patchelf.

    This replaces all existing RPATH entries with the standard ROCm RPATH,
    producing a clean shared library suitable for wheel packaging.

    The name should include the .so extension (e.g. "my_plugin.so")
    and is used as both the target name and output filename.

    Args:
        name: Name of the target and output file (e.g. "my_plugin.so").
        src: Label of the input shared library.
        visibility: Target visibility.
    """
    native.genrule(
        name = name,
        srcs = [src],
        outs = [name],
        cmd = "cp $(location {}) $@ && patchelf --set-rpath '{}' $@".format(src, _ROCM_RPATH),
        visibility = visibility,
    )
