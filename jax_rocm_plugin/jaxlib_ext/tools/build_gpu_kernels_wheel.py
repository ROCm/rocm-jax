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

# Most users should not run this script directly; use build.py instead.
# pylint: disable=duplicate-code

"""
Script to build a JAX ROCm kernel plugin wheel. Intended for use via Bazel.
"""

import argparse
import functools
import os
import pathlib
import stat
import subprocess
import tempfile


# pylint: disable=import-error,invalid-name,consider-using-with
from bazel_tools.tools.python.runfiles import runfiles
from jaxlib_ext.tools import build_utils

parser = argparse.ArgumentParser()
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
    "--rocm_jax_git_hash",
    default="",
    required=True,
    help="Git hash. Empty if unknown. Optional.",
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
    required=True,
    help="rocm/xla Git hash. Empty if unknown. Optional.",
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
    required=True,
    help="rocm/jax Git hash. Empty if unknown. Optional.",
)

args = parser.parse_args()

r = runfiles.Create()
PYEXT = "pyd" if build_utils.is_windows() else "so"


def write_setup_cfg(setup_sources_path, cpu):
    """Write setup.cfg file for wheel build."""
    tag = build_utils.platform_tag(cpu)
    cfg_path = setup_sources_path / "setup.cfg"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(
            f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
"""
        )


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


def prepare_wheel_rocm(wheel_sources_path: pathlib.Path, *, cpu, rocm_version):
    """Assembles a source tree for the rocm kernel wheel in `sources_path`."""
    copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

    copy_runfiles(
        "__main__/jax_plugins/rocm/plugin_pyproject.toml",
        dst_dir=wheel_sources_path,
        dst_filename="pyproject.toml",
    )
    copy_runfiles(
        "__main__/jax_plugins/rocm/plugin_setup.py",
        dst_dir=wheel_sources_path,
        dst_filename="setup.py",
    )
    build_utils.update_setup_with_rocm_version(wheel_sources_path, rocm_version)
    write_setup_cfg(wheel_sources_path, cpu)
    plugin_dir = wheel_sources_path / f"jax_rocm{rocm_version}_plugin"
    xla_commit_hash = get_xla_commit_hash()
    jax_commit_hash = get_jax_commit_hash()
    build_utils.write_commit_info(
        plugin_dir, xla_commit_hash, jax_commit_hash, args.rocm_jax_git_hash
    )
    copy_runfiles(
        dst_dir=plugin_dir,
        src_files=[
            f"jax/jaxlib/rocm/_linalg.{PYEXT}",
            f"jax/jaxlib/rocm/_prng.{PYEXT}",
            f"jax/jaxlib/rocm/_solver.{PYEXT}",
            f"jax/jaxlib/rocm/_sparse.{PYEXT}",
            f"jax/jaxlib/rocm/_hybrid.{PYEXT}",
            f"jax/jaxlib/rocm/_rnn.{PYEXT}",
            f"jax/jaxlib/rocm/_triton.{PYEXT}",
            f"jax/jaxlib/rocm/rocm_plugin_extension.{PYEXT}",
            "jax/jaxlib/version.py",
        ],
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

    files = [
        f"_linalg.{PYEXT}",
        f"_prng.{PYEXT}",
        f"_solver.{PYEXT}",
        f"_sparse.{PYEXT}",
        f"_hybrid.{PYEXT}",
        f"_rnn.{PYEXT}",
        f"_triton.{PYEXT}",
        f"rocm_plugin_extension.{PYEXT}",
    ]
    runpath = "$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib:/opt/rocm/lib"
    # patchelf --force-rpath --set-rpath $RUNPATH $so
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


def main():
    """Main entry point for building the ROCm plugin wheel."""
    sources_path = args.sources_path
    with tempfile.TemporaryDirectory(prefix="jax_rocm_plugin") as tmpdir:

        if sources_path is None:
            sources_path = tmpdir

        os.makedirs(args.output_path, exist_ok=True)

        if args.enable_rocm:
            prepare_wheel_rocm(
                pathlib.Path(sources_path),
                cpu=args.cpu,
                rocm_version=args.platform_version,
            )
            package_name = f"jax rocm{args.platform_version} plugin"
        else:
            raise ValueError("Unsupported backend. Choose 'rocm'.")

        if args.editable:
            build_utils.build_editable(sources_path, args.output_path, package_name)
        else:
            git_hash = build_utils.get_githash(args.rocm_jax_git_hash)
            build_utils.build_wheel(
                sources_path,
                args.output_path,
                package_name,
                git_hash=git_hash,
            )


if __name__ == "__main__":
    main()
