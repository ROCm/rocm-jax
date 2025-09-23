#!/usr/bin/env python3
"""
Script to build and fix JAX ROCm plugin and PJRT wheels.
"""

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


# NOTE(mrodden): This file is part of the ROCm build scripts, and
# needs be compatible with Python 3.6. Please do not include these
# in any "upgrade" scripts


import argparse
from collections import deque
import fcntl
import logging
import os
import re
import select
import subprocess
import sys


LOG = logging.getLogger(__name__)


GPU_DEVICE_TARGETS = (
    "gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1200 gfx1201"
)


def get_rocm_version(rocm_path):
    try:
        version = subprocess.check_output(
            f"cat {rocm_path}/.info/version | cut -d '-' -f 1", shell=True
        )
        return version.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching ROCm version: {e}")
        return None


def build_rocm_path(rocm_version_str):
    """Return appropriate ROCm installation path."""
    path = "/opt/rocm-%s" % rocm_version_str
    if os.path.exists(path):
        return path
    return os.path.realpath("/opt/rocm")


def update_rocm_targets(rocm_path, targets):
    """
    Writes the list of GPU targets to bin/target.lst under the given ROCm path
    (excluding gfx950 for ROCm < 7.0.0, since XLA doesn't support it),
    and mimics 'touch' on .info/version to signal updates.

    Args:
        rocm_path (str): The root ROCm installation directory.
        targets (str): A space-separated string of GPU targets.
    """
    target_fp = os.path.join(rocm_path, "bin/target.lst")
    version_fp = os.path.join(rocm_path, ".info/version")

    try:
        with open(version_fp, "r", encoding="utf-8") as f:
            version = re.search(r"\d+\.\d+\.\d+", f.read()).group(0)
    except (FileNotFoundError, OSError):
        version = "0.0.0"

    def supports_gfx950(v):
        return tuple(map(int, v.split("."))) >= (7, 0, 0)

    filtered = [t for t in targets.split() if t != "gfx950" or supports_gfx950(version)]

    # Write targets one per line.
    with open(target_fp, "w", encoding="utf-8") as fd:
        fd.write("\n".join(filtered) + "\n")

    # mimic touch
    # pylint: disable=R1732
    open(version_fp, "a", encoding="utf-8").close()


def find_clang_path():
    """Search for and return the best clang binary path."""
    llvm_base_path = "/usr/lib/"
    # Search for llvm directories and pick the highest version.
    llvm_dirs = [d for d in os.listdir(llvm_base_path) if d.startswith("llvm-")]
    if llvm_dirs:
        # Sort to get the highest llvm version.
        llvm_dirs.sort(reverse=True)
        clang_bin_dir = os.path.join(llvm_base_path, llvm_dirs[0], "bin")

        # Prefer versioned clang binaries (e.g., clang-18).
        versioned_clang = None
        generic_clang = None

        for f in os.listdir(clang_bin_dir):
            # Checks for versioned clang binaries.
            if f.startswith("clang-") and f[6:].isdigit():
                versioned_clang = os.path.join(clang_bin_dir, f)
            # Fallback to non-versioned clang.
            elif f == "clang":
                generic_clang = os.path.join(clang_bin_dir, f)

        # Return versioned clang if available, otherwise return generic clang.
        if versioned_clang:
            return versioned_clang
        if generic_clang:
            return generic_clang

    return None


# pylint: disable=R0913, R0917
def build_jaxlib_wheel(
    jax_path, rocm_path, rocm_version, python_version, output_dir, xla_path=None, compiler="gcc"
):
    """Build jaxlib and ROCm plugin wheels."""
    use_clang = compiler == "clang"

    # Avoid git warning by setting safe.directory.
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", "*"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to configure Git safe directory: {e}")
        raise

    version_string = rocm_version[0]
    if version_string == "6":
        version_string = "60"

    cmd = [
        "python",
        "build/build.py",
        "build",
        "--wheels=jax-rocm-plugin,jax-rocm-pjrt",
        "--rocm_path=%s" % rocm_path,
        "--rocm_version=%s" % version_string,
        "--use_clang=%s" % use_clang,
        "--verbose",
        "--output_path=%s" % output_dir,
    ]

    # Add clang path if clang is used.
    if use_clang:
        clang_path = find_clang_path()
        if clang_path:
            cmd.append("--clang_path=%s" % clang_path)
        else:
            raise RuntimeError("Clang binary not found in /usr/lib/llvm-*")

    if xla_path:
        cmd.append("--bazel_options=--override_repository=xla=%s" % xla_path)

    cpy = to_cpy_ver(python_version)
    py_bin = "/opt/python/%s-%s/bin" % (cpy, cpy)

    env = dict(os.environ)
    env["JAX_RELEASE"] = str(1)
    env["JAXLIB_RELEASE"] = str(1)
    env["PATH"] = "%s:%s" % (py_bin, env["PATH"])

    LOG.info("Running %r from cwd=%r", cmd, jax_path)
    pattern = re.compile("Output wheel: (.+)\n")

    _run_scan_for_output(cmd, pattern, env=env, cwd=jax_path, capture="stderr")


# pylint: disable=R0914
def _run_scan_for_output(cmd, pattern, env=None, cwd=None, capture=None):
    """Run subprocess and scan output for regex match."""
    buf = deque(maxlen=20000)

    popen_args = {
        "args": cmd,
        "env": env,
        "cwd": cwd,
        "stdout": subprocess.PIPE if capture != "stderr" else None,
        "stderr": subprocess.PIPE if capture == "stderr" else None,
    }

    with subprocess.Popen(**popen_args) as p:
        cap_fd = p.stderr if capture == "stderr" else p.stdout
        redir = sys.stderr if capture == "stderr" else sys.stdout

        flags = fcntl.fcntl(cap_fd, fcntl.F_GETFL)
        fcntl.fcntl(cap_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        eof = False
        while not eof:
            r, _, _ = select.select([cap_fd], [], [])
            for fd in r:
                dat = fd.read(512)
                if dat:
                    t = dat.decode("utf8")
                    redir.write(t)
                    buf.extend(t)
                else:
                    eof = True

        p.communicate()

        if p.returncode != 0:
            raise RuntimeError(
                "Child process exited with nonzero result: rc=%d" % p.returncode
            )

    text = "".join(buf)
    matches = pattern.findall(text)

    if not matches:
        LOG.error("No wheel name found in output: %r", text)
        raise RuntimeError("No wheel name found in output")

    wheels = []
    for match in matches:
        LOG.info("Found built wheel: %r", match)
        wheels.append(match)

    return wheels


def to_cpy_ver(python_version):
    """Convert Python version string (e.g., 3.10) to CPython tag (e.g., cp310)."""
    tup = python_version.split(".")
    return "cp%d%d" % (int(tup[0]), int(tup[1]))


def fix_wheel(path, jax_path):
    """Fix auditwheel compliance using fixwheel.py and auditwheel."""
    try:
        # NOTE(mrodden): fixwheel needs auditwheel 6.0.0, which has a min python of 3.8
        # so use one of the CPythons in /opt to run
        env = dict(os.environ)
        py_bin = "/opt/python/cp310-cp310/bin"
        env["PATH"] = "%s:%s" % (py_bin, env["PATH"])

        # NOTE(mrodden): auditwheel 6.0 added lddtree module, but 6.3.0 changed
        # the fuction to ldd and also changed its behavior
        # constrain range to 6.0 to 6.2.x
        cmd = ["pip", "install", "auditwheel>=6,<6.3", "wheel"]
        subprocess.run(cmd, check=True, env=env)

        fixwheel_path = os.path.join(jax_path, "build/rocm/tools/fixwheel.py")
        cmd = ["python", fixwheel_path, path]
        subprocess.run(cmd, check=True, env=env)
        LOG.info("Wheel fix completed successfully.")
    except subprocess.CalledProcessError as cpe:
        LOG.error("Subprocess failed with error: %s", cpe)
        raise
    except Exception as e:
        LOG.error("An unexpected error occurred: %s", e)
        raise


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    rocm_spec = p.add_mutually_exclusive_group()
    rocm_spec.add_argument(
        "--rocm-version", help="ROCM Version to build JAX against"
    )
    rocm_spec.add_argument(
        "rocm-path", help="Path to ROCM to build JAX against"
    )
    p.add_argument(
        "--python-versions",
        default=["3.10.19,3.12"],
        help="Comma separated CPython versions that wheels will be built and output for",
    )
    p.add_argument(
        "--xla-path",
        type=str,
        default=None,
        help="Optional directory where XLA source is located to use instead of JAX builtin XLA",
    )
    p.add_argument(
        "--compiler",
        type=str,
        default="gcc",
        help="Compiler backend to use when compiling jax/jaxlib",
    )

    p.add_argument("jax_path", help="Directory where JAX source directory is located")

    return p.parse_args()


def find_wheels(path):
    """Return list of wheel files in given path."""
    wheels = []

    for f in os.listdir(path):
        if f.endswith(".whl"):
            wheels.append(os.path.join(path, f))

    LOG.info("Found wheels: %r", wheels)
    return wheels


def main():
    """Main entry point."""
    args = parse_args()
    python_versions = args.python_versions.split(",")

    manylinux_output_dir = "dist_manylinux"

    rocm_path=args.rocm_path
    if args.rocm_version:
        rocm_path = build_rocm_path(args.rocm_version)
        rocm_version = args.rocm_version
    else:
        if rocm_path is None:
            rocm_path = "/opt/rocm"
        rocm_version=get_rocm_version(rocm_path)

    print("ROCM_PATH=%s" % rocm_path)
    print("ROCM_VERSION=%s" % rocm_version)
    print("PYTHON_VERSIONS=%r" % python_versions)
    print("JAX_PATH=%s" % args.jax_path)
    print("XLA_PATH=%s" % args.xla_path)
    print("COMPILER=%s" % args.compiler)
    print("OUTPUT_DIR=%s" % manylinux_output_dir)

    update_rocm_targets(rocm_path, GPU_DEVICE_TARGETS)

    full_output_path = os.path.join(args.jax_path, manylinux_output_dir)
    os.makedirs(full_output_path, exist_ok=True)

    # wipe anything in output dir before building new
    wheel_paths = find_wheels(full_output_path)
    for whl in wheel_paths:
        print("Removing wheel=%r" % whl)
        os.remove(whl)

    for py in python_versions:
        build_jaxlib_wheel(
            args.jax_path,
            rocm_path,
            rocm_version,
            py,
            full_output_path,
            args.xla_path,
            args.compiler,
        )
        wheel_paths = find_wheels(full_output_path)
        for wheel_path in wheel_paths:
            fix_wheel(wheel_path, args.jax_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
