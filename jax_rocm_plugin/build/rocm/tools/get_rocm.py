#!/usr/bin/env python3

# Copyright 2024 The JAX Authors.
# Copyright 2025 Mathew Odden <mathewrodden@gmail.com>
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

"""Script for installing ROCm from various places"""

import argparse
import json
import logging
import os
import sys
import subprocess
import urllib.request


# pylint: disable=unspecified-encoding
LOG = logging.getLogger(__name__)


class RocmInstallException(Exception):
    """Exceptions thrown when trying to install ROCm"""


def latest_rocm():
    """
    Retrieve and return a version of the newest release from repo.radeon.com

    Returns a string of the form X.Y.Z
    """
    with urllib.request.urlopen(
        "https://api.github.com/repos/rocm/rocm/releases/latest"
    ) as rocm_releases:
        dat = rocm_releases.read()
        rd = json.loads(dat)
        _, ver_str = rd["tag_name"].split("-")
        return ver_str


def os_release_meta():
    """Read /etc/os-release metadata and return as key-value pairs."""
    try:
        with open("/etc/os-release") as rel_file:
            os_rel = rel_file.read()

            kvs = {}
            for line in os_rel.split("\n"):
                if line.strip():
                    k, v = line.strip().split("=", 1)
                    v = v.strip('"')
                    kvs[k] = v
            return kvs
    except OSError:
        return None


# pylint: disable=useless-object-inheritance
class System(object):
    """
    Class to abstract the package manager and other
    OS dependent operations.
    """

    def __init__(self, pkgbin, rocm_package_list):
        self.pkgbin = pkgbin
        self.rocm_package_list = rocm_package_list

    def install_packages(self, package_specs):
        """Install packages from a list of specifications, i.e. ['wget'. 'cowsay>6.0']"""
        
        # Update package lists before installing
        env = dict(os.environ)
        if self.pkgbin == "apt":
            env["DEBIAN_FRONTEND"] = "noninteractive"
            update_cmd = [self.pkgbin, "update"]
            LOG.info("Running %r", update_cmd)
            subprocess.check_call(update_cmd, env=env)
        elif self.pkgbin == "dnf":
            update_cmd = [self.pkgbin, "makecache"]
            LOG.info("Running %r", update_cmd)
            subprocess.check_call(update_cmd, env=env)

        cmd = [
            self.pkgbin,
            "install",
            "-y",
        ]
        cmd.extend(package_specs)

        LOG.info("Running %r", cmd)
        subprocess.check_call(cmd, env=env)

    def install_rocm(self):
        """Install ROCm from this System's package list"""
        self.install_packages(self.rocm_package_list)


UBUNTU = System(
    pkgbin="apt",
    rocm_package_list=[
        "rocm-dev",
        "rocm-libs",
    ],
)


RHEL8 = System(
    pkgbin="dnf",
    rocm_package_list=[
        "libdrm-amdgpu",
        "rocm-dev",
        "rocm-ml-sdk",
        "miopen-hip ",
        "miopen-hip-devel",
        "rocblas",
        "rocblas-devel",
        "rocsolver-devel",
        "rocrand-devel",
        "rocfft-devel",
        "hipfft-devel",
        "hipblas-devel",
        "rocprim-devel",
        "hipcub-devel",
        "rccl-devel",
        "hipsparse-devel",
        "hipsolver-devel",
    ],
)


def parse_version(version_str):
    """
    Parse a ROCm version string into a Version type.

    >>> print(parse_version("1.2.3"))
    ... Version(major=1, minor=2, rev=3)
    """
    if isinstance(version_str, str):
        parts = version_str.split(".")
        rv = type("Version", (), {})()
        rv.major = int(parts[0].strip())
        rv.minor = int(parts[1].strip())
        rv.rev = None

        if len(parts) > 2:
            rv.rev = int(parts[2].strip())

    else:
        rv = version_str

    return rv


def get_system():
    """
    Factory function for System instances.

    Returns a System object for the current host OS type.
    """
    md = os_release_meta()

    if md["ID"] == "ubuntu":
        return UBUNTU

    if md["ID"] in ["almalinux", "rhel", "fedora", "centos"]:
        if md["PLATFORM_ID"] == "platform:el8":
            return RHEL8

    raise RocmInstallException("No system for %r" % md)


def _setup_internal_repo(system, rocm_version, job_name, build_num):
    """Set up repos for getting internal ROCm builds"""
    # wget is required by amdgpu-repo
    system.install_packages(["wget"])

    install_amdgpu_installer_internal(rocm_version)

    amdgpu_build = None
    with urllib.request.urlopen(
        "http://rocm-ci.amd.com/job/%s/%s/artifact/amdgpu_kernel_info.txt"
        % (job_name, build_num)
    ) as kernel_info:
        amdgpu_build = kernel_info.read().decode("utf8").strip()

    cmd = [
        "amdgpu-repo",
        "--amdgpu-build=%s" % amdgpu_build,
        "--rocm-build=%s/%s" % (job_name, build_num),
    ]
    LOG.info("Running %r", cmd)
    subprocess.check_call(cmd)

    cmd = [
        "amdgpu-install",
        "--no-dkms",
        "--usecase=rocm",
        "-y",
    ]

    env = dict(os.environ)
    if system.pkgbin == "apt":
        env["DEBIAN_FRONTEND"] = "noninteractive"

    LOG.info("Running %r", cmd)
    subprocess.check_call(cmd, env=env)


def install_rocm(rocm_version, job_name=None, build_num=None):
    """Download and install the requested version of ROCm."""

    s = get_system()

    if job_name and build_num:
        _setup_internal_repo(s, rocm_version, job_name, build_num)
    else:
        if s == RHEL8:
            setup_repos_el8(rocm_version)
        elif s == UBUNTU:
            setup_repos_ubuntu(rocm_version)
        else:
            raise RocmInstallException("Platform not supported")

    s.install_rocm()


def install_amdgpu_installer_internal(rocm_version):
    """
    Download and install the "amdgpu-installer" package from internal builds
    on the current system.
    """
    md = os_release_meta()
    url, fn = _build_installer_url(rocm_version, md)

    try:
        # download installer
        LOG.info("Downloading from %s", url)
        urllib.request.urlretrieve(url, filename=fn)

        system = get_system()

        cmd = [system.pkgbin, "install", "-y", "./%s" % fn]
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass


def _build_installer_url(rocm_version, metadata):
    """Build the URL to the amdgpu installer for your ROCm version and OS"""
    md = metadata

    rv = parse_version(rocm_version)

    base_url = "https://artifactory-cdn.amd.com/artifactory/list"

    if md["ID"] == "ubuntu":
        fmt = "amdgpu-install-internal_%(rocm_major)s.%(rocm_minor)s-%(os_version)s-1_all.deb"
        package_name = fmt % {
            "rocm_major": rv.major,
            "rocm_minor": rv.minor,
            "os_version": md["VERSION_ID"],
        }

        url = "%s/amdgpu-deb/%s" % (base_url, package_name)
    elif md.get("PLATFORM_ID") == "platform:el8":
        fmt = "amdgpu-install-internal-%(rocm_major)s.%(rocm_minor)s_%(os_version)s-1.noarch.rpm"
        package_name = fmt % {
            "rocm_major": rv.major,
            "rocm_minor": rv.minor,
            "os_version": "8",
        }

        url = "%s/amdgpu-rpm/rhel/%s" % (base_url, package_name)
    else:
        raise RocmInstallException("Platform not supported: %r" % md)

    return url, package_name


APT_RADEON_PIN_CONTENT = """
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
"""


def setup_repos_ubuntu(rocm_version_str):
    """Configure an apt sources list entry for ROCm."""

    rv = parse_version(rocm_version_str)

    # if X.Y.0 -> repo url version should be X.Y
    if rv.rev == 0:
        rocm_version_str = "%d.%d" % (rv.major, rv.minor)

    # update indexes before prereq install, for fresh docker images
    subprocess.check_call(["apt-get", "update"])

    s = get_system()
    s.install_packages(["wget", "sudo", "gnupg"])

    md = os_release_meta()
    codename = md["VERSION_CODENAME"]

    keyadd = "wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -"
    subprocess.check_call(keyadd, shell=True)

    with open("/etc/apt/sources.list.d/amdgpu.list", "w") as fd:
        fd.write(
            ("deb [arch=amd64] " "https://repo.radeon.com/amdgpu/%s/ubuntu %s main\n")
            % (rocm_version_str, codename)
        )

    with open("/etc/apt/sources.list.d/rocm.list", "w") as fd:
        fd.write(
            ("deb [arch=amd64] " "https://repo.radeon.com/rocm/apt/%s %s main\n")
            % (rocm_version_str, codename)
        )

    # on ubuntu 22 or greater, debian community rocm packages
    # conflict with repo.radeon.com packages
    with open("/etc/apt/preferences.d/rocm-pin-600", "w") as fd:
        fd.write(APT_RADEON_PIN_CONTENT)

    # update indexes after new repo install
    subprocess.check_call(["apt-get", "update"])


def setup_repos_el8(rocm_version_str):
    """Configure a yum repo entry for ROCm."""

    rv = parse_version(rocm_version_str)

    # if X.Y.0 -> repo url version should be X.Y
    if rv.rev == 0:
        rocm_version_str = "%d.%d" % (rv.major, rv.minor)

    with open("/etc/yum.repos.d/rocm.repo", "w") as rfd:
        rfd.write(
            """
[ROCm]
name=ROCm
baseurl=http://repo.radeon.com/rocm/rhel8/%s/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
"""
            % rocm_version_str
        )

    with open("/etc/yum.repos.d/amdgpu.repo", "w") as afd:
        afd.write(
            """
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/%s/rhel/8.8/main/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
"""
            % rocm_version_str
        )


def parse_args():
    """Parse command-line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--rocm-version", help="ROCm version to install", default="latest")
    p.add_argument("--job-name", default=None)
    p.add_argument("--build-num", default=None)
    return p.parse_args()


def main():
    """Installs ROCm at /opt/rocm on your system"""
    args = parse_args()
    if args.rocm_version == "latest":
        try:
            rocm_version = latest_rocm()
            print("Latest ROCm release: %s" % rocm_version)
        # pylint: disable=W0718
        except Exception:
            print(
                "Latest ROCm lookup failed. Please use '--rocm-version' to specify a "
                "version instead.",
                file=sys.stderr,
            )
            sys.exit(-1)
    else:
        rocm_version = args.rocm_version

    install_rocm(rocm_version, job_name=args.job_name, build_num=args.build_num)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
