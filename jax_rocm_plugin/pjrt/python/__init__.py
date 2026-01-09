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

"""JAX ROCm plugin initialization module."""

import functools
import importlib
import logging
import os
import os.path
import pathlib
import re

from jax._src.lib import xla_client  # pylint: disable=import-error
import jax._src.xla_bridge as xb  # pylint: disable=import-error

# rocm_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax rocm plugin packages.
for pkg_name in ["jax_rocm7_plugin", "jax_rocm60_plugin", "jaxlib.rocm"]:
    try:
        rocm_plugin_extension = importlib.import_module(
            f"{pkg_name}.rocm_plugin_extension"
        )
    except ImportError:
        rocm_plugin_extension = None  # pylint: disable=invalid-name
    else:
        break

logger = logging.getLogger(__name__)


def _get_library_path():
    base_path = pathlib.Path(__file__).resolve().parent
    installed_path = base_path / "xla_rocm_plugin.so"
    if installed_path.exists():
        return installed_path

    local_path = base_path / "pjrt_c_api_gpu_plugin.so"
    if not local_path.exists():
        runfiles_dir = os.getenv("RUNFILES_DIR", None)
        if runfiles_dir:
            local_path = pathlib.Path(
                os.path.join(runfiles_dir, "xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so")
            )

    if local_path.exists():
        logger.debug(
            "Native library %s does not exist. This most likely indicates an issue"
            " with how %s was built or installed. Fallback to local test"
            " library %s",
            installed_path,
            __package__,
            local_path,
        )
        return local_path

    logger.debug(
        "WARNING: Native library %s and local test library path %s do not"
        " exist. This most likely indicates an issue with how %s was built or"
        " installed or missing src files.",
        installed_path,
        local_path,
        __package__,
    )
    return None


def set_rocm_paths(path):  # pylint: disable=too-many-branches
    """Set ROCm environment paths for bitcode and linker.

    Args:
        path: Path to the ROCm plugin library.
    """
    rocm_lib = None
    try:
        import rocm  # pylint: disable=import-outside-toplevel

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

    logger.info("ROCm wheel install found at %r", rocm_lib)

    bitcode_path = ""
    lld_path = ""

    for root, _dirs, files in os.walk(os.path.join(rocm_lib, "llvm")):
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


def is_amd_gpu_available() -> bool:
    """Checks if any AMD GPU is available to safeguard loading of the ROCm pjrt.

    Different approaches to this are possible. We chose testing for existence
    of KFD kernel driver entities as a proxy for the presence of AMD GPUs as
    a good compromise between performance, reliability and simplicity.
    Presence of such entities doesn't guarantee that the GPUs are usable through
    HIP and PJRT, however, we can't do much much better without spawning an
    additional process with a potentially complicated setup to run actual HIP
    code. And we don't want to initialize HIP right now inside the current
    process, because doing so might spoil a proper initialization of the
    rocprofiler-sdk later during PJRT startup."""

    try:
        kfd_nodes_path = "/sys/class/kfd/kfd/topology/nodes/"
        if not os.path.exists(kfd_nodes_path):
            return False

        # the RE matches strings like "simd_count ##" and extracts the number ##
        r_simd_count = re.compile(r"\bsimd_count\s+(\d+)\b", re.MULTILINE)
        # we're using a non-zero simd_count as a trait of a GPU following the
        # KFD implementation
        # https://github.com/torvalds/linux/blob/ea1013c1539270e372fc99854bc6e4d94eaeff66/drivers/gpu/drm/amd/amdkfd/kfd_topology.c#L941

        for node in os.listdir(kfd_nodes_path):
            node_props_path = os.path.join(kfd_nodes_path, node, "properties")
            if not os.path.exists(node_props_path):
                continue

            try:
                file_size = os.path.getsize(node_props_path)
                # 16KB is more than a reasonable limit
                if file_size <= 0 or file_size > 16 * 1024:
                    continue

                with open(node_props_path, "r", encoding="ascii") as f:
                    match = r_simd_count.search(f.read())
                    if match:
                        simd_count = int(match.group(1))
                        if simd_count > 0:
                            return True  # one is enough
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "Failed to read KFD node file '%s': %s", node_props_path, e
                )
                continue

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to check for AMD KFD presence: %s", e)
    return False


def initialize():
    """Initialize the JAX ROCm plugin."""
    path = _get_library_path()
    if path is None:
        return

    set_rocm_paths(path)

    if rocm_plugin_extension is None:
        logger.warning("rocm_plugin_extension not found")
        return

    if not is_amd_gpu_available():
        raise ValueError("No AMD GPUs were found, skipping ROCm plugin initialization")

    options = xla_client.generate_pjrt_gpu_plugin_options()
    options["platform_name"] = "ROCM"
    c_api = xb.register_plugin(
        "rocm", priority=500, library_path=str(path), options=options
    )
    if rocm_plugin_extension:
        xla_client.register_custom_call_handler(
            "ROCM",
            functools.partial(rocm_plugin_extension.register_custom_call_target, c_api),
        )
        for _name, _value in rocm_plugin_extension.ffi_registrations().items():
            xla_client.register_custom_call_target(
                _name, _value, platform="ROCM", api_version=1
            )

        xla_client.register_custom_type_id_handler(
            "ROCM",
            functools.partial(rocm_plugin_extension.register_custom_type_id, c_api),
        )
    else:
        logger.warning("rocm_plugin_extension is not found.")
