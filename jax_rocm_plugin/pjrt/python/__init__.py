# Copyright 2024 The JAX Authors.
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

import functools
import importlib
import logging
import os
import pathlib
import re
import warnings

from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

# rocm_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax rocm plugin packages.
for pkg_name in ['jax_rocm7_plugin', 'jax_rocm60_plugin', 'jaxlib.rocm']:
  try:
    rocm_plugin_extension = importlib.import_module(
        f'{pkg_name}.rocm_plugin_extension'
    )
  except ImportError:
    rocm_plugin_extension = None
  else:
    break

logger = logging.getLogger(__name__)


def _get_jax_version():
  """Get JAX version for version checking."""
  try:
    import jax
    return jax.__version__
  except ImportError:
    logger.warning("Could not import jax to check version compatibility")
    return None


def _get_pjrt_wheel_version():
  """Get the current PJRT wheel version."""
  try:
    from . import version
    return version.__version__
  except ImportError:
    logger.warning("Could not import PJRT wheel version module")
    return None


def _check_pjrt_wheel_version(pjrt_wheel_name: str, jax_version: str, pjrt_version: str) -> bool:
  """Check if PJRT wheel version is compatible with JAX version.

  This implements the same logic as jaxlib.plugin_support.check_plugin_version
  for ROCm PJRT wheels.
  """
  # Regex to match a dotted version prefix 0.1.23.456.789 of a PEP440 version.
  version_regex = re.compile(r"[0-9]+(?:\.[0-9]+)*")

  def _parse_version(v: str) -> tuple[int, ...]:
    m = version_regex.match(v)
    if m is None:
      raise ValueError(f"Unable to parse version string '{v}'")
    return tuple(int(x) for x in m.group(0).split("."))

  try:
    jax_parsed = _parse_version(jax_version)
    pjrt_parsed = _parse_version(pjrt_version)
  except ValueError as e:
    logger.warning(f"Failed to parse version strings: {e}")
    return True  # Allow loading if we can't parse versions

  # Check if version checking is disabled via environment variable
  if os.environ.get("JAX_DEBUG_SKIP_ROCM_PLUGIN_VERSION_CHECK", "").lower() in ("1", "true", "yes"):
    warnings.warn(
        f"JAX ROCm PJRT wheel {pjrt_wheel_name} version checking has been disabled via "
        "JAX_DEBUG_SKIP_ROCM_PLUGIN_VERSION_CHECK environment variable. "
        "This may lead to compatibility issues.",
        RuntimeWarning,
    )
    return True

  # For ROCm PJRT wheels: major versions must match, PJRT wheel minor version must be <= JAX minor version
  if len(jax_parsed) < 2 or len(pjrt_parsed) < 2:
    # If either version doesn't have major.minor format, fall back to exact match
    compatible = jax_parsed == pjrt_parsed
  else:
    # Check major version match and minor version constraint
    major_match = jax_parsed[0] == pjrt_parsed[0]
    minor_compatible = pjrt_parsed[1] <= jax_parsed[1]
    compatible = major_match and minor_compatible

  if not compatible:
    warnings.warn(
        f"JAX ROCm PJRT wheel {pjrt_wheel_name} version {pjrt_version} is installed, but "
        "it is not compatible with the installed JAX version "
        f"{jax_version}. ROCm PJRT wheels require matching major versions and "
        "PJRT wheel minor version <= JAX minor version, so it will not be used. "
        f"Use JAX_DEBUG_SKIP_ROCM_PLUGIN_VERSION_CHECK=1 to override",
        RuntimeWarning,
    )
    return False
  return True


def _get_library_path():
  base_path = pathlib.Path(__file__).resolve().parent
  installed_path = (
      base_path / 'xla_rocm_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = (
      base_path / 'pjrt_c_api_gpu_plugin.so'
  )
  if not local_path.exists():
    runfiles_dir = os.getenv('RUNFILES_DIR', None)
    if runfiles_dir:
      local_path = pathlib.Path(
          os.path.join(runfiles_dir, 'xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so')
      )

  if local_path.exists():
    logger.debug(
        'Native library %s does not exist. This most likely indicates an issue'
        ' with how %s was built or installed. Fallback to local test'
        ' library %s',
        installed_path,
        __package__,
        local_path,
    )
    return local_path

  logger.debug(
      'WARNING: Native library %s and local test library path %s do not'
      ' exist. This most likely indicates an issue with how %s was built or'
      ' installed or missing src files.',
      installed_path,
      local_path,
      __package__,
  )
  return None


def set_rocm_paths(path):
  rocm_lib = None
  try:
    import rocm
    rocm_lib = os.path.join(rocm.__path__[0], "lib")
  except ImportError:
    # find python site-packages
    sp = path.parent.parent.parent
    maybe_rocm_lib = os.path.join(sp, "rocm/lib")
    if os.path.exists(maybe_rocm_lib):
      rocm_lib = maybe_rocm_lib

  if not rocm_lib:
    logger.info("No ROCm wheel installation found")
    return
  else:
    logger.info("ROCm wheel install found at %r" % rocm_lib)

  bitcode_path = ""
  lld_path = ""

  for root, dirs, files in os.walk(os.path.join(rocm_lib, "llvm")):
    # look for ld.lld and ocml.bc
    for f in files:
      if f == "ocml.bc":
        bitcode_path = root
      if f == "ld.lld":
        # amd backend needs the directory not the full path to binary
        lld_path = root

    if bitcode_path and lld_path:
      break


  if not bitcode_path:
    logger.warning("jax_rocm_plugin couldn't locate amdgpu bitcode")
  else:
    logger.info("jax_rocm_plugin using bitcode found at %r", bitcode_path)

  if not lld_path:
    logger.warning("jax_rocm_plugin couldn't locate amdgpu ld.lld")
  else:
    logger.info("jax_rocm_plugin using ld.lld found at %r", lld_path)

  os.environ["JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH"] = bitcode_path
  os.environ["HIP_DEVICE_LIB_PATH"] = bitcode_path
  os.environ["JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH"] = lld_path


def initialize():
  path = _get_library_path()
  if path is None:
    return

  set_rocm_paths(path)

  # Version compatibility check
  jax_version = _get_jax_version()
  pjrt_version = _get_pjrt_wheel_version()

  if jax_version and pjrt_version:
    pjrt_wheel_name = __package__ or "jax_rocm_pjrt"
    if not _check_pjrt_wheel_version(pjrt_wheel_name, jax_version, pjrt_version):
      logger.warning(f"Skipping ROCm PJRT plugin initialization due to version incompatibility")
      return
    logger.info(f"ROCm PJRT wheel version {pjrt_version} is compatible with JAX version {jax_version}")
  else:
    logger.warning("Could not perform PJRT wheel version check, proceeding with plugin initialization")

  options = xla_client.generate_pjrt_gpu_plugin_options()
  options["platform_name"] = "ROCM"
  c_api = xb.register_plugin(
      'rocm', priority=500, library_path=str(path), options=options
  )
  if rocm_plugin_extension:
    xla_client.register_custom_call_handler(
        "ROCM",
        functools.partial(
            rocm_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in rocm_plugin_extension.ffi_registrations().items():
      xla_client.register_custom_call_target(_name, _value, platform="ROCM", api_version=1)

    xla_client.register_custom_type_id_handler(
        "ROCM",
        functools.partial(
            rocm_plugin_extension.register_custom_type_id, c_api
        ),
    )
  else:
    logger.warning('rocm_plugin_extension is not found.')
