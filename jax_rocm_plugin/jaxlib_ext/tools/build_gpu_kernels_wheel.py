# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script that builds a jax-rocm-plugin wheel for ROCm kernels.

This script is intended to be run via bazel run as part of the JAX ROCm plugin
build process. Most users should not run this script directly; use build.py instead.
"""

import argparse
import functools
import os
import pathlib
import shutil
import stat
import subprocess
import tempfile

# pylint: disable=import-error,invalid-name,consider-using-with
from bazel_tools.tools.python.runfiles import runfiles
from jaxlib_ext.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--jaxlib_git_hash",
    default="",
    help="Git hash passed by jax_wheel macro. Empty if unknown.",
)
parser.add_argument(
    "--rocm_jax_git_hash", default="", help="Git hash. Empty if unknown."
)
parser.add_argument(
    "--srcs", action="append", help="Source files (passed by jax_wheel macro, unused)."
)
parser.add_argument(
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--platform_version",
    default=None,
    required=True,
    help="Target CUDA/ROCM version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' jax cuda/rocm plugin build instead of a wheel.",
)
parser.add_argument(
    "--enable-cuda",
    default=False,
    help="Should we build with CUDA enabled? Requires CUDA and CuDNN.",
)
parser.add_argument(
    "--enable-rocm", default=False, help="Should we build with ROCM enabled?"
)

parser.add_argument(
    "--xla-commit", default="", help="rocm/xla Git hash. Empty if unknown."
)

parser.add_argument(
    "--use_local_xla",
    type=str,
    default="",
    help="Path to local XLA repository. If not set, uses pinned commit hash",
)

parser.add_argument(
    "--use_local_jax",
    type=str,
    default="",
    help="Path to local JAX repository. If not set, uses pinned commit hash",
)

parser.add_argument(
    "--jax-commit", default="", help="rocm/jax Git hash. Empty if unknown."
)

args = parser.parse_args()


def get_rocm_jax_git_hash():
    """Get git hash, preferring rocm_jax_git_hash, falling back to jaxlib_git_hash."""
    return args.rocm_jax_git_hash or args.jaxlib_git_hash or ""


r = runfiles.Create()
pyext = "pyd" if build_utils.is_windows() else "so"


def write_setup_cfg(setup_sources_path, cpu):
    """Write setup.cfg file for wheel build."""
    tag = build_utils.platform_tag(cpu)
    cfg_path = setup_sources_path / "setup.cfg"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(f"""[metadata]
                    license_files = LICENSE.txt
                    [bdist_wheel]
                    plat_name={tag}
                 """)


def get_xla_commit_hash():
    """Determines the XLA commit hash to use - local repository or a pinned."""
    if args.use_local_xla:
        return build_utils.get_local_git_commit(args.use_local_xla)

    print(f"Using pinned XLA commit hash: {args.xla_commit}")
    return args.xla_commit


def get_jax_commit_hash():
    """Determines the JAX commit hash to use - local repository or a pinned."""
    if args.use_local_jax:
        return build_utils.get_local_git_commit(args.use_local_jax)

    print(f"Using pinned JAX commit hash: {args.jax_commit}")
    return args.jax_commit


def build_source_map(srcs):
    """Build a map from basename to full path for source files."""
    source_map = {}
    if srcs:
        for src in srcs:
            basename = os.path.basename(src)
            source_map[basename] = src
    return source_map


def copy_from_srcs_or_runfiles(source_map, src_basename, dst_dir, dst_filename=None):
    """Copy a file from --srcs or fall back to runfiles."""
    dst_filename = dst_filename or src_basename
    dst_path = os.path.join(dst_dir, dst_filename)
    os.makedirs(dst_dir, exist_ok=True)

    # Try --srcs first
    if src_basename in source_map:
        shutil.copy(source_map[src_basename], dst_path)
        return

    # Fall back to runfiles with various prefixes
    for prefix in ["__main__/", "jax_rocm_plugin/", "jax/", ""]:
        runfile_path = r.Rlocation(f"{prefix}{src_basename}")
        if runfile_path and os.path.exists(runfile_path):
            shutil.copy(runfile_path, dst_path)
            return

    raise FileNotFoundError(f"Unable to find source file: {src_basename}")


# pylint: disable=too-many-locals
def prepare_wheel_rocm(wheel_sources_path: pathlib.Path, cpu, rocm_version, srcs):
    """Assembles a source tree for the rocm kernel wheel in `sources_path`."""
    source_map = build_source_map(srcs)
    copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

    # Copy pyproject.toml and setup.py
    copy_from_srcs_or_runfiles(
        source_map, "plugin_pyproject.toml", wheel_sources_path, "pyproject.toml"
    )
    copy_from_srcs_or_runfiles(
        source_map, "plugin_setup.py", wheel_sources_path, "setup.py"
    )

    build_utils.update_setup_with_rocm_version(wheel_sources_path, rocm_version)
    write_setup_cfg(wheel_sources_path, cpu)
    plugin_dir = wheel_sources_path / f"jax_rocm{rocm_version}_plugin"
    os.makedirs(plugin_dir, exist_ok=True)

    xla_commit_hash = get_xla_commit_hash()
    jax_commit_hash = get_jax_commit_hash()
    build_utils.write_commit_info(
        plugin_dir, xla_commit_hash, jax_commit_hash, get_rocm_jax_git_hash()
    )

    # Copy version.py
    copy_from_srcs_or_runfiles(source_map, "version.py", plugin_dir)

    # Copy shared libraries - these come from @jax//jaxlib/rocm via runfiles
    so_files = [
        f"_linalg.{pyext}",
        f"_prng.{pyext}",
        f"_solver.{pyext}",
        f"_sparse.{pyext}",
        f"_hybrid.{pyext}",
        f"_rnn.{pyext}",
        f"_triton.{pyext}",
        f"rocm_plugin_extension.{pyext}",
    ]
    for so_file in so_files:
        # Try source map first, then runfiles with jax/jaxlib/rocm prefix
        if so_file in source_map:
            copy_from_srcs_or_runfiles(source_map, so_file, plugin_dir)
        else:
            copy_runfiles(f"jax/jaxlib/rocm/{so_file}", dst_dir=plugin_dir)

    # NOTE(mrodden): this is a hack to change/set rpath values
    # in the shared objects that are produced by the bazel build
    # before they get pulled into the wheel build process.
    # we have to do this change here because setting rpath
    # using bazel requires the rpath to be valid during the build
    # which won't be correct until we make changes to
    # the xla/tsl/jax plugin build

    try:
        subprocess.check_output(["which", "patchelf"])
    except subprocess.CalledProcessError as ex:
        mesg = (
            "rocm plugin and kernel wheel builds require patchelf. "
            "please install 'patchelf' and run again"
        )
        raise RuntimeError(mesg) from ex

    files = [
        f"_linalg.{pyext}",
        f"_prng.{pyext}",
        f"_solver.{pyext}",
        f"_sparse.{pyext}",
        f"_hybrid.{pyext}",
        f"_rnn.{pyext}",
        f"_triton.{pyext}",
        f"rocm_plugin_extension.{pyext}",
    ]
    runpath = "$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib:/opt/rocm/lib"
    # patchelf --set-rpath $RUNPATH $so
    for f in files:
        so_path = os.path.join(plugin_dir, f)
        fix_perms = False
        perms = os.stat(so_path).st_mode
        if not perms & stat.S_IWUSR:
            fix_perms = True
            os.chmod(so_path, perms | stat.S_IWUSR)
        subprocess.check_call(["patchelf", "--set-rpath", runpath, so_path])
        if fix_perms:
            os.chmod(so_path, perms)


tmpdir = tempfile.TemporaryDirectory(prefix="jax_rocm_plugin")
sources_path = tmpdir.name
try:
    os.makedirs(args.output_path, exist_ok=True)
    prepare_wheel_rocm(
        pathlib.Path(sources_path),
        cpu=args.cpu,
        rocm_version=args.platform_version,
        srcs=args.srcs,
    )
    package_name = f"jax rocm{args.platform_version} plugin"
    if args.editable:
        build_utils.build_editable(sources_path, args.output_path, package_name)
    else:
        git_hash = build_utils.get_githash(get_rocm_jax_git_hash())
        build_utils.build_wheel(
            sources_path, args.output_path, package_name, git_hash=git_hash
        )
finally:
    tmpdir.cleanup()
