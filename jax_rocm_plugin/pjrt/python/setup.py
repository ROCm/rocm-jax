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
"""Setup script for the ROCm JAX PJRT plugin package.

Holds only lightweight metadata wiring; build/version logic resides in the
generated version helper inside the package hierarchy.
"""

import importlib
import os
from setuptools import setup, find_namespace_packages

__version__ = None
rocm_version = 0  # placeholder (runtime substituted)  # pylint: disable=invalid-name
project_name = f"jax-rocm{rocm_version}-pjrt"
package_name = f"jax_plugins.xla_rocm{rocm_version}"

# Extract ROCm version from the `ROCM_PATH` environment variable.
DEFAULT_ROCM_PATH = "/opt/rocm"
rocm_path = os.getenv("ROCM_PATH", DEFAULT_ROCM_PATH)
rocm_detected_version = rocm_path.split("-")[-1] if "-" in rocm_path else "7.0"


def load_version_module(pkg_path):
    """Import and return the package's version helper module dynamically."""
    spec = importlib.util.spec_from_file_location(
        "version", os.path.join(pkg_path, "version.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_version_module = load_version_module(f"jax_plugins/xla_rocm{rocm_version}")
__version__ = _version_module._get_version_for_build()  # pylint: disable=protected-access

packages = find_namespace_packages(include=[package_name, f"{package_name}.*"])

setup(
    name=project_name,
    version=__version__,
    description=f"JAX XLA PJRT Plugin for AMD GPUs (ROCm:{rocm_detected_version})",
    long_description="",
    long_description_content_type="text/markdown",
    author="GulsumGA-AMD",
    author_email="Gulsum.GudukbayAkbulut@amd.com",
    packages=packages,
    install_requires=[],
    url="https://github.com/ROCm/rocm-jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
    ],
    package_data={
        package_name: ["xla_rocm_plugin.so"],
    },
    zip_safe=False,
    entry_points={
        "jax_plugins": [
            f"xla_rocm{rocm_version} = {package_name}",
        ],
    },
)
