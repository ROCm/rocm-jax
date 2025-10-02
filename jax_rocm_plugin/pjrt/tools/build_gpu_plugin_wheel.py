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

# Script that builds a jax cuda/rocm plugin wheel, intended to be run via bazel run
# as part of the jax cuda/rocm plugin build process.

# Most users should not run this script directly; use build.py instead.

"""
Script to build a JAX ROCm plugin wheel. Intended for use via Bazel.
"""

import argparse
import functools
import os
import pathlib
import stat
import subprocess
import tempfile

# pylint: disable=import-error
from bazel_tools.tools.python.runfiles import runfiles
from pjrt.tools import build_utils


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
    "--jaxlib_git_hash",
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
    "--enable-rocm", default=False, help="Should we build with ROCM enabled?"
)
args = parser.parse_args()

r = runfiles.Create()


def write_setup_cfg(setup_cfg_path, cpu):
    """Write setup.cfg with platform tag."""
    tag = build_utils.platform_tag(cpu)
    with open(setup_cfg_path / "setup.cfg", "w", encoding="utf-8") as cfg_file:
        cfg_file.write(
            f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
python-tag=py3
"""
        )


def prepare_rocm_plugin_wheel(rocm_sources_path: pathlib.Path, *, cpu, rocm_version):
    """Assembles a source tree for the ROCm wheel in `rocm_sources_path`."""
    copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

    plugin_dir = rocm_sources_path / "jax_plugins" / f"xla_rocm{rocm_version}"
    copy_runfiles(
        dst_dir=rocm_sources_path,
        src_files=[
            "__main__/pjrt/python/pyproject.toml",
            "__main__/pjrt/python/setup.py",
        ],
    )
    build_utils.update_setup_with_rocm_version(rocm_sources_path, rocm_version)
    write_setup_cfg(rocm_sources_path, cpu)
    copy_runfiles(
        dst_dir=plugin_dir,
        src_files=[
            "__main__/pjrt/python/__init__.py",
            "__main__/pjrt/python/version.py",
        ],
    )
    copy_runfiles(
        "__main__/pjrt/pjrt_c_api_gpu_plugin.so",
        dst_dir=plugin_dir,
        dst_filename="xla_rocm_plugin.so",
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


def main():
    """Main entry point for building the ROCm plugin wheel."""
    sources_path = args.sources_path
    with tempfile.TemporaryDirectory(prefix="jaxgpupjrt") as tmpdir:
        if sources_path is None:
            sources_path = tmpdir
        os.makedirs(args.output_path, exist_ok=True)

        if args.enable_rocm:
            prepare_rocm_plugin_wheel(
                pathlib.Path(sources_path),
                cpu=args.cpu,
                rocm_version=args.platform_version,
            )
            package_name = "jax rocm plugin"
        else:
            raise ValueError("Unsupported backend. Choose 'rocm'.")

        if args.editable:
            build_utils.build_editable(sources_path, args.output_path, package_name)
        else:
            git_hash = build_utils.get_githash(args.jaxlib_git_hash)
            build_utils.build_wheel(
                sources_path,
                args.output_path,
                package_name,
                git_hash=git_hash,
            )


if __name__ == "__main__":
    main()
