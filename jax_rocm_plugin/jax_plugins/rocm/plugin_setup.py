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

"""Setup script for JAX ROCm plugin package."""

import importlib
import os
import re
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
rocm_version = 0  # placeholder  # pylint: disable=invalid-name
gpu_arch = ""  # placeholder  # pylint: disable=invalid-name
_arch_suffix = f"-{gpu_arch}" if gpu_arch else ""  # pylint: disable=invalid-name
_arch_uscore = f"_{gpu_arch}" if gpu_arch else ""  # pylint: disable=invalid-name
project_name = (  # pylint: disable=invalid-name
    f"jax-rocm{rocm_version}-plugin{_arch_suffix}"
)
package_name = (  # pylint: disable=invalid-name
    f"jax_rocm{rocm_version}_plugin{_arch_uscore}"
)

# Extract ROCm version from the `ROCM_PATH` environment variable.
default_rocm_path = "/opt/rocm"  # pylint: disable=invalid-name
rocm_path = os.getenv("ROCM_PATH", default_rocm_path)
rocm_tag = os.getenv("ROCM_VERSION_EXTRA")


def detect_rocm_version(path, tag):
    """Detect ROCm version from env tag, ROCM_VERSION env, rocm path, or pip."""
    if tag:
        return tag
    rocm_ver_env = os.getenv("ROCM_VERSION")
    if rocm_ver_env:
        return rocm_ver_env
    for candidate in (path, os.path.realpath(path)):
        match = re.search(r"(\d+(?:\.\d+)+)", candidate)
        if match:
            return match.group(1)
    try:
        import subprocess  # pylint: disable=import-outside-toplevel

        result = subprocess.run(
            ["pip", "list"], capture_output=True, text=True, check=False
        )
        for line in result.stdout.splitlines():
            if line.startswith("rocm "):
                return line.split()[-1]
    except Exception:  # pylint: disable=broad-except
        pass
    return "unknown"


rocm_detected_version = detect_rocm_version(rocm_path, rocm_tag)
ROCM_PROVIDER = (
    "TheRock" if os.getenv("THEROCK_BUILD") else "Legacy ROCm"
)  # pylint: disable=invalid-name


def load_version_module(pkg_path):
    """Load version module from the given package path.

    Args:
        pkg_path: Path to the package containing version.py

    Returns:
        The loaded version module
    """
    spec = importlib.util.spec_from_file_location(
        "version", os.path.join(pkg_path, "version.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_version_module = load_version_module(package_name)
__version__ = (
    _version_module._get_version_for_build()  # pylint: disable=protected-access
)
if rocm_tag:
    __version__ = __version__ + "+rocm" + rocm_tag
_cmdclass = _version_module._get_cmdclass(  # pylint: disable=protected-access
    package_name
)


class BinaryDistribution(Distribution):
    """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

    def has_ext_modules(self):
        """Indicate that this distribution has extension modules.

        Returns:
            bool: Always True to include ABI tag
        """
        return True


setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description=f"JAX Plugin for AMD GPUs (ROCm: {ROCM_PROVIDER} {rocm_detected_version})",
    long_description="",
    long_description_content_type="text/markdown",
    author="ROCm JAX Devs",
    author_email="dl.dl-JAX@amd.com",
    packages=[package_name],
    python_requires=">=3.11",
    install_requires=[
        f"jax-rocm{rocm_version}-pjrt{_arch_suffix}=="
        f"{_version_module._version}.*"  # pylint: disable=protected-access
    ],
    url="https://github.com/ROCm/rocm-jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    package_data={
        package_name: [
            "*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    extras_require={
        "with_rocm": [
            "amd_rocm_hip_runtime_devel_instinct",
            "amd_rocm_hip_runtime_instinct",
            "amd_hipblas_instinct",
            "amd_hipsparse_instinct",
            "amd_hipsolver_instinct",
            "amd_miopen_hip_instinct",
            "amd_rocm_llvm_instinct",
            "amd_rocm_language_runtime_instinct",
            "amd_rccl_instinct",
            "amd_hipfft_instinct",
            "amd_rocm_device_libs_instinct",
            "amd_hipsparselt_instinct",
        ],
    },
)
