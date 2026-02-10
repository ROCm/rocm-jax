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
    "--rocm_jax_git_hash",
    default="",
    help="Git hash. Empty if unknown.",
)
parser.add_argument(
    "--srcs",
    action="append",
    help="Source files passed by jax_wheel macro. If provided, these are used "
    "for config files. .so files always come from runfiles.",
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
    "--xla-commit",
    default="",
    help="rocm/xla Git hash. Empty if unknown.",
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
    "--jax-commit",
    default="",
    help="rocm/jax Git hash. Empty if unknown.",
)

args = parser.parse_args()


def get_rocm_jax_git_hash():
    """Get git hash, preferring rocm_jax_git_hash, falling back to jaxlib_git_hash."""
    return args.rocm_jax_git_hash or args.jaxlib_git_hash or ""


r = runfiles.Create()
pyext = "pyd" if build_utils.is_windows() else "so"


def rloc(path):
    """Get runfiles location, trying multiple workspace prefixes."""
    for prefix in ["__main__", "jax_rocm_plugin"]:
        loc = r.Rlocation(f"{prefix}/{path}")
        if loc is not None:
            return loc
    raise FileNotFoundError(f"Unable to find in runfiles: {path}")


def find_src(srcs, basename):
    """Find a file in srcs by basename."""
    for src in srcs:
        if os.path.basename(src) == basename:
            return src
    raise FileNotFoundError(f"'{basename}' not found in --srcs")


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


def prepare_wheel_rocm(wheel_sources_path: pathlib.Path, *, cpu, rocm_version, srcs):
    # pylint: disable=too-many-locals
    """Assembles a source tree for the rocm kernel wheel in `sources_path`."""
    plugin_dir = wheel_sources_path / f"jax_rocm{rocm_version}_plugin"
    os.makedirs(plugin_dir, exist_ok=True)

    # Copy config files: from --srcs if provided, else from runfiles
    if srcs:
        shutil.copy(
            find_src(srcs, "plugin_pyproject.toml"),
            wheel_sources_path / "pyproject.toml",
        )
        shutil.copy(find_src(srcs, "plugin_setup.py"), wheel_sources_path / "setup.py")
        shutil.copy(find_src(srcs, "LICENSE.txt"), wheel_sources_path)
        shutil.copy(find_src(srcs, "version.py"), plugin_dir)
    else:
        shutil.copy(
            rloc("jax_plugins/rocm/plugin_pyproject.toml"),
            wheel_sources_path / "pyproject.toml",
        )
        shutil.copy(
            rloc("jax_plugins/rocm/plugin_setup.py"), wheel_sources_path / "setup.py"
        )
        shutil.copy(rloc("jaxlib_ext/tools/LICENSE.txt"), wheel_sources_path)
        shutil.copy(rloc("pjrt/python/version.py"), plugin_dir)

    build_utils.update_setup_with_rocm_version(wheel_sources_path, rocm_version)
    write_setup_cfg(wheel_sources_path, cpu)
    xla_commit_hash = get_xla_commit_hash()
    jax_commit_hash = get_jax_commit_hash()
    build_utils.write_commit_info(
        plugin_dir, xla_commit_hash, jax_commit_hash, get_rocm_jax_git_hash()
    )

    # Copy .so files from local wrapper targets (//jaxlib_ext/rocm).
    # RPATHs are set at build time via Bazel features/linkopts when
    # --config=rocm_wheel is used (rocm_path_type=link_only).
    for so_file in [
        f"_linalg.{pyext}",
        f"_prng.{pyext}",
        f"_solver.{pyext}",
        f"_sparse.{pyext}",
        f"_hybrid.{pyext}",
        f"_rnn.{pyext}",
        f"_triton.{pyext}",
        f"rocm_plugin_extension.{pyext}",
    ]:
        shutil.copy(rloc(f"jaxlib_ext/rocm/{so_file}"), plugin_dir)


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
            sources_path,
            args.output_path,
            package_name,
            git_hash=git_hash,
        )
finally:
    tmpdir.cleanup()
