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

# Script that builds a jax-cuda12-plugin wheel for cuda kernels, intended to be
# run via bazel run as part of the jax cuda plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import stat
import subprocess
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jaxlib_ext.tools import build_utils

parser = argparse.ArgumentParser()
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
    "--enable-cuda",
    default=False,
    help="Should we build with CUDA enabled? Requires CUDA and CuDNN.")
parser.add_argument(
    "--enable-rocm",
    default=False,
    help="Should we build with ROCM enabled?")
args = parser.parse_args()

r = runfiles.Create()
pyext = "pyd" if build_utils.is_windows() else "so"


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
""")


def prepare_wheel_rocm(
    sources_path: pathlib.Path, *, cpu, rocm_version
):
  """Assembles a source tree for the rocm kernel wheel in `sources_path`."""
  copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

  copy_runfiles(
      "__main__/jax_plugins/rocm/plugin_pyproject.toml",
      dst_dir=sources_path,
      dst_filename="pyproject.toml",
  )
  copy_runfiles(
      "__main__/jax_plugins/rocm/plugin_setup.py",
      dst_dir=sources_path,
      dst_filename="setup.py",
  )
  build_utils.update_setup_with_rocm_version(sources_path, rocm_version)
  write_setup_cfg(sources_path, cpu)

  plugin_dir = sources_path / f"jax_rocm{rocm_version}_plugin"
  copy_runfiles(
      dst_dir=plugin_dir,
      src_files=[
          f"jax/jaxlib/rocm/_linalg.{pyext}",
          f"jax/jaxlib/rocm/_prng.{pyext}",
          f"jax/jaxlib/rocm/_solver.{pyext}",
          f"jax/jaxlib/rocm/_sparse.{pyext}",
          f"jax/jaxlib/rocm/_hybrid.{pyext}",
          f"jax/jaxlib/rocm/_rnn.{pyext}",
          f"jax/jaxlib/rocm/_triton.{pyext}",
          f"jax/jaxlib/rocm/rocm_plugin_extension.{pyext}",
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
    raise Exception(mesg) from ex

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
  runpath = '$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib'
  # patchelf --force-rpath --set-rpath $RUNPATH $so
  for f in files:
    so_path = os.path.join(plugin_dir, f)
    fix_perms = False
    perms = os.stat(so_path).st_mode
    if not perms & stat.S_IWUSR:
        fix_perms = True
        os.chmod(so_path, perms | stat.S_IWUSR)
    subprocess.check_call(["patchelf", "--force-rpath", "--set-rpath", runpath, so_path])
    if fix_perms:
        os.chmod(so_path, perms)

tmpdir = tempfile.TemporaryDirectory(prefix="jax_rocm_plugin")
sources_path = tmpdir.name
try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel_rocm(
      pathlib.Path(sources_path), cpu=args.cpu, rocm_version=args.platform_version
  )
  package_name = f"jax rocm{args.platform_version} plugin"
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
finally:
  tmpdir.cleanup()
