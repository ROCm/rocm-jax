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

"""Utilities for the building JAX related python packages."""

from __future__ import annotations

import os
import pathlib
import platform
import shutil
import sys
import subprocess
import glob
from collections.abc import Sequence
import textwrap


def is_windows() -> bool:
    """Check if running on Windows platform."""
    return sys.platform.startswith("win32")


def platform_tag(cpu: str) -> str:
    """Generate platform-specific wheel tag based on CPU architecture."""
    # Match the platform tags from @jax//jaxlib:jax.bzl PLATFORM_TAGS_DICT
    platform_name, cpu_name = {
        ("Linux", "x86_64"): ("manylinux_2_27", "x86_64"),
        ("Linux", "aarch64"): ("manylinux_2_27", "aarch64"),
        ("Linux", "ppc64le"): ("manylinux_2_27", "ppc64le"),
        ("Darwin", "x86_64"): ("macosx_11_0", "x86_64"),
        ("Darwin", "arm64"): ("macosx_11_0", "arm64"),
        ("Windows", "AMD64"): ("win", "amd64"),
    }[(platform.system(), cpu)]
    return f"{platform_name}_{cpu_name}"


def get_githash(jaxlib_git_hash):
    """Get git hash from file or return the hash directly."""
    if jaxlib_git_hash != "" and os.path.isfile(jaxlib_git_hash):
        with open(jaxlib_git_hash, "r", encoding="utf-8") as f:
            return f.readline().strip()
    return jaxlib_git_hash


def build_wheel(
    sources_path: str, output_path: str, package_name: str, git_hash: str = ""
) -> None:
    """Builds a wheel in `output_path` using the source tree in `sources_path`."""
    env = dict(os.environ)
    if git_hash:
        env["JAX_GIT_HASH"] = git_hash
    subprocess.run(
        [sys.executable, "-m", "build", "-n", "-w"],
        check=True,
        cwd=sources_path,
        env=env,
    )
    for wheel in glob.glob(os.path.join(sources_path, "dist", "*.whl")):
        output_file = os.path.join(output_path, os.path.basename(wheel))
        sys.stderr.write(f"Output wheel: {output_file}\n\n")
        sys.stderr.write(
            f"To install the newly-built {package_name} wheel "
            + "on system Python, run:\n"
        )
        sys.stderr.write(f"  pip install {output_file} --force-reinstall\n\n")

        py_version = ".".join(platform.python_version_tuple()[:-1])
        sys.stderr.write(
            f"To install the newly-built {package_name} wheel "
            + "on hermetic Python, run:\n"
        )
        sys.stderr.write(f'  echo -e "\\n{output_file}" >> build/requirements.in\n')
        sys.stderr.write(
            "  bazel run //build:requirements.update"
            + f" --repo_env=HERMETIC_PYTHON_VERSION={py_version}\n\n"
        )
        shutil.copy(wheel, output_path)


def build_editable(sources_path: str, output_path: str, package_name: str) -> None:
    """self-explanatory"""
    sys.stderr.write(
        f"To install the editable {package_name} build, run:\n\n"
        f"  pip install -e {output_path}\n\n"
    )
    shutil.rmtree(output_path, ignore_errors=True)
    shutil.copytree(sources_path, output_path)


def update_setup_with_cuda_version(file_dir: pathlib.Path, cuda_version: str):
    """Update setup.py with the specified CUDA version."""
    src_file = file_dir / "setup.py"
    with open(src_file, encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        "cuda_version = 0  # placeholder", f"cuda_version = {cuda_version}"
    )
    with open(src_file, "w", encoding="utf-8") as f:
        f.write(content)


def update_setup_with_rocm_version(file_dir: pathlib.Path, rocm_version: str):
    """Update setup.py with the specified ROCm version."""
    src_file = file_dir / "setup.py"
    with open(src_file, encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        "rocm_version = 0  # placeholder", f"rocm_version = {rocm_version}"
    )
    with open(src_file, "w", encoding="utf-8") as f:
        f.write(content)


def write_commit_info(plugin_dir, xla_commit, jax_commit, rocm_jax_commit):
    """Write commit hash information into commit_info.py inside `plugin_dir`."""
    os.makedirs(plugin_dir, exist_ok=True)
    commit_info_content = textwrap.dedent(f"""
        # auto-generated; do not edit

        commits = {{
            "ROCm/xla": "{xla_commit}",
            "ROCm/rocm-jax": "{rocm_jax_commit}",
            "jax": "{jax_commit}",
        }}
    """)

    commit_info_path = plugin_dir / "commit_info.py"

    with open(commit_info_path, "w", encoding="utf-8") as f:
        f.write(commit_info_content)


def get_local_git_commit(repo_path):
    """Get commit hash from local git repository."""
    git_dir = os.path.join(repo_path, ".git")
    if os.path.exists(git_dir):
        try:
            result = subprocess.run(
                ["git", "--git-dir", git_dir, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            commit_hash = result.stdout.strip()
            print(f"Successfully retrieved commit hash from {repo_path}: {commit_hash}")
            return commit_hash
        except subprocess.CalledProcessError as e:
            print(f"Failed to get commit from {repo_path}: {e}")

    raise RuntimeError("Could not retrieve commit info, failing build..")
