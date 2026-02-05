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
import os
import pathlib
import shutil
import stat
import subprocess
import tempfile

# pylint: disable=import-error,invalid-name,consider-using-with
from bazel_tools.tools.python.runfiles import runfiles
from pjrt.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--sources_path",
    default=None,
    help="Path in which the wheel's sources should be prepared. Optional. If "
    "omitted, a temporary directory will be used.",
)
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
    "--rocm_jax_git_hash", default="", help="rocm-jax Git hash. Empty if unknown."
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


def build_source_map(srcs):
    """Build a map from basename to full path for source files."""
    source_map = {}
    if srcs:
        for src in srcs:
            basename = os.path.basename(src)
            source_map[basename] = src
    return source_map


def copy_from_srcs_or_runfiles(source_map, src_path, dst_dir, dst_filename=None):
    """Copy a file from --srcs or fall back to runfiles.
    
    Args:
        source_map: Map from basename to full path for --srcs files
        src_path: Path or basename of the source file (e.g., "pjrt/python/version.py")
        dst_dir: Destination directory
        dst_filename: Filename to use in destination (defaults to basename of src_path)
    """
    src_basename = os.path.basename(src_path)
    dst_filename = dst_filename or src_basename
    dst_path = os.path.join(dst_dir, dst_filename)
    os.makedirs(dst_dir, exist_ok=True)

    # Try --srcs first (by basename)
    if src_basename in source_map:
        shutil.copy(source_map[src_basename], dst_path)
        return

    # Fall back to runfiles with various prefixes
    # Try the full path first, then with workspace prefixes
    paths_to_try = [
        src_path,  # As provided (e.g., "pjrt/python/version.py")
        f"jax_rocm_plugin/{src_path}",
        f"__main__/{src_path}",
        src_basename,  # Just basename
        f"jax_rocm_plugin/{src_basename}",
        f"__main__/{src_basename}",
    ]
    for path in paths_to_try:
        runfile_path = r.Rlocation(path)
        if runfile_path and os.path.exists(runfile_path):
            shutil.copy(runfile_path, dst_path)
            return

    raise FileNotFoundError(f"Unable to find source file: {src_path}")


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


def prepare_rocm_plugin_wheel(
    wheel_sources_path: pathlib.Path, *, cpu, rocm_version, srcs
):
    """Assembles a source tree for the ROCm wheel in `sources_path`."""
    source_map = build_source_map(srcs)

    plugin_dir = wheel_sources_path / "jax_plugins" / f"xla_rocm{rocm_version}"

    # Copy pyproject.toml, setup.py, and LICENSE.txt
    copy_from_srcs_or_runfiles(source_map, "pjrt/python/pyproject.toml", wheel_sources_path)
    copy_from_srcs_or_runfiles(source_map, "pjrt/python/setup.py", wheel_sources_path)
    copy_from_srcs_or_runfiles(source_map, "pjrt/tools/LICENSE.txt", wheel_sources_path)
    build_utils.update_setup_with_rocm_version(wheel_sources_path, rocm_version)
    write_setup_cfg(wheel_sources_path, cpu)
    xla_commit_hash = get_xla_commit_hash()
    jax_commit_hash = get_jax_commit_hash()
    build_utils.write_commit_info(
        plugin_dir, xla_commit_hash, jax_commit_hash, get_rocm_jax_git_hash()
    )

    # Copy plugin files
    copy_from_srcs_or_runfiles(source_map, "pjrt/python/__init__.py", plugin_dir)
    copy_from_srcs_or_runfiles(source_map, "pjrt/python/version.py", plugin_dir)

    # Copy the PJRT plugin .so file
    copy_from_srcs_or_runfiles(
        source_map, "pjrt/pjrt_c_api_gpu_plugin.so", plugin_dir, "xla_rocm_plugin.so"
    )

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

    shared_obj_path = os.path.join(plugin_dir, "xla_rocm_plugin.so")
    runpath = "$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib:/opt/rocm/lib"
    # patchelf --set-rpath $RUNPATH $so
    fix_perms = False
    perms = os.stat(shared_obj_path).st_mode
    if not perms & stat.S_IWUSR:
        fix_perms = True
        os.chmod(shared_obj_path, perms | stat.S_IWUSR)
    subprocess.check_call(["patchelf", "--set-rpath", runpath, shared_obj_path])
    if fix_perms:
        os.chmod(shared_obj_path, perms)


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
    tmpdir = tempfile.TemporaryDirectory(prefix="jaxgpupjrt")
    sources_path = tmpdir.name

try:
    os.makedirs(args.output_path, exist_ok=True)

    if args.enable_rocm:
        prepare_rocm_plugin_wheel(
            pathlib.Path(sources_path),
            cpu=args.cpu,
            rocm_version=args.platform_version,
            srcs=args.srcs,
        )
        package_name = "jax rocm plugin"
    else:
        raise ValueError("Unsupported backend. Choose 'rocm'.")

    if args.editable:
        build_utils.build_editable(sources_path, args.output_path, package_name)
    else:
        git_hash = build_utils.get_githash(get_rocm_jax_git_hash())
        build_utils.build_wheel(
            sources_path, args.output_path, package_name, git_hash=git_hash
        )
finally:
    if tmpdir:
        tmpdir.cleanup()
