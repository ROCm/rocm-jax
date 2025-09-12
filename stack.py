#!/usr/bin/env python3
"""Script for setting up local development environments"""

import argparse
import os
import subprocess


TEST_JAX_REPO_REF = "rocm-jaxlib-v0.6.0"
XLA_REPO_REF = "rocm-jaxlib-v0.6.0"


JAX_REPL_URL = "https://github.com/rocm/jax"
XLA_REPL_URL = "https://github.com/rocm/xla"

DEFAULT_XLA_DIR = "../xla"
DEFAULT_KERNELS_JAX_DIR = "../jax"

PLUGIN_NAMESPACE_VERSION = "7"


MAKE_TEMPLATE = r"""
# gfx targets for which XLA and jax custom call kernels are built for
# AMDGPU_TARGETS ?= "gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201"

# customize to a single arch for local dev builds to reduce compile time
AMDGPU_TARGETS ?= "$(shell rocminfo | grep -o -m 1 'gfx.*')"

# Defines a value for '--bazel_options' for each of 3 build types (pjrt, plugin + jaxlib).
# By default, uses local XLA for each wheel. Redefine to whatever option is needed for your case
ALL_BAZEL_OPTIONS="--override_repository=xla=%(xla_dir)s"

# Use your local JAX for building the kernels in jax_rocm_plugin
# KERNELS_JAX_OVERRIDE_OPTION="--override_repository=jax=../jax"
KERNELS_JAX_OVERRIDE_OPTION="%(kernels_jax_override)s"

.PHONY: test clean install dist

.default: dist


dist: jax_rocm_plugin jax_rocm_pjrt


jax_rocm_plugin:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-plugin \
            --target_cpu_features=native \
            --rocm_path=/opt/rocm/ \
            --rocm_version=%(plugin_version)s \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
            --bazel_options=${ALL_BAZEL_OPTIONS} \
            --bazel_options=${KERNELS_JAX_OVERRIDE_OPTION} \
            --verbose \
            --clang_path=%(clang_path)s


jax_rocm_pjrt:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-pjrt \
            --target_cpu_features=native \
            --rocm_path=/opt/rocm/ \
            --rocm_version=%(plugin_version)s \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
            --bazel_options=${ALL_BAZEL_OPTIONS} \
            --bazel_options=${KERNELS_JAX_OVERRIDE_OPTION} \
            --verbose \
            --clang_path=%(clang_path)s


clean:
	rm -rf dist


install: dist
	pip install --force-reinstall dist/*


refresh: clean dist install


test:
	python3 tests/test_plugin.py


# Sometimes developers might want to build their own jaxlib. Usually, we can
# just use the one from upstream, but we might want to build our own if we
# suspect that jaxlib isn't loading the plugin properly or if ROCm-specific
# code is somehow making its way into jaxlib.

jaxlib:
	(cd %(kernels_jax_dir)s && python3 ./build/build.py build \
            --target_cpu_features=native \
            --use_clang=true \
            --clang_path=%(clang_path)s \
            --wheels=jaxlib \
            --bazel_options=${ALL_BAZEL_OPTIONS} \
            --verbose \
	)


jaxlib_clean:
	rm -f %(kernels_jax_dir)s/dist/*


jaxlib_install:
	pip install --force-reinstall %(kernels_jax_dir)s/dist/*


refresh_jaxlib: jaxlib_clean jaxlib jaxlib_install
"""


def find_clang():
    """Find a local clang compiler and return its file path."""

    clang_path = None

    # check PATH
    try:
        out = subprocess.check_output(["which", "clang"])
        clang_path = out.decode("utf-8").strip()
        return clang_path
    except subprocess.CalledProcessError:
        pass

    # search /usr/lib/
    top = "/usr/lib"
    for root, dirs, files in os.walk(top):

        # only walk llvm dirs
        if root == top:
            for d in dirs:
                if not d.startswith("llvm"):
                    dirs.remove(d)

        for f in files:
            if f == "clang":
                clang_path = os.path.join(root, f)
                return clang_path

    # We didn't find a clang install
    return None


def setup_development(
    xla_ref: str,
    xla_dir: str,
    test_jax_ref: str,
    kernels_jax_dir: str,
    rebuild_makefile: bool = False,
):
    """Clone jax and xla repos, and set up Makefile for developers"""

    # Always clone the JAX repo that we'll use for running unit tests
    if not os.path.exists("./jax"):
        cmd = ["git", "clone"]
        cmd.extend(["--branch", test_jax_ref])
        cmd.append(JAX_REPL_URL)
        subprocess.check_call(cmd)

    # clone xla from source for building jax_rocm_plugin if the user didn't
    # specify an existing XLA directory
    if not os.path.exists("./xla") and xla_dir == DEFAULT_XLA_DIR:
        cmd = ["git", "clone"]
        cmd.extend(["--branch", xla_ref])
        cmd.append(XLA_REPL_URL)
        subprocess.check_call(cmd)

    # create build/install/test script
    makefile_path = "./jax_rocm_plugin/Makefile"
    if rebuild_makefile or not os.path.exists(makefile_path):
        kvs = {
            "clang_path": "/usr/lib/llvm-18/bin/clang",
            "plugin_version": PLUGIN_NAMESPACE_VERSION,
            "xla_dir": xla_dir,
            # If the user wants to use their own JAX for building the plugin wheel
            # that contains all the jaxlib kernel code (jax_rocm7_plugin), add that
            # to the Makefile.
            "kernels_jax_override": (
                ("--override_repository=jax=%s" % kernels_jax_dir)
                if kernels_jax_dir
                else ""
            ),
            "kernels_jax_dir": kernels_jax_dir if kernels_jax_dir else "",
        }

        clang_path = find_clang()
        if clang_path:
            print("Found clang at %r" % clang_path)
            kvs["clang_path"] = clang_path
        else:
            print("No clang found. Defaulting to %r" % kvs["clang_path"])

        makefile_content = MAKE_TEMPLATE % kvs

        with open(makefile_path, "w", encoding="utf-8") as mf:
            mf.write(makefile_content)


def dev_docker(rm):
    """Start a docker container for local plugin development"""
    cur_abs_path = os.path.abspath(os.curdir)
    image_name = "ubuntu:22.04"

    ep = "/rocm-jax/tools/docker_dev_setup.sh"

    cmd = [
        "docker",
        "run",
        "-it",
        "--network=host",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--ipc=host",
        "--shm-size=16G",
        "--group-add",
        "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        "%s:/rocm-jax" % cur_abs_path,
        "--env",
        "ROCM_JAX_DIR=/rocm-jax",
        "--env",
        "_IS_ENTRYPOINT=1",
        "--entrypoint=%s" % ep,
    ]

    if rm:
        cmd.append("--rm")

    cmd.append(image_name)

    with subprocess.Popen(cmd) as p:
        p.wait()


# build mode setup


# install jax/jaxlib from known versions
# setup build/install/test script
def setup_build():
    """Setup for building the plugin locally"""
    raise NotImplementedError


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()

    subp = p.add_subparsers(dest="action", required=True)

    dev = subp.add_parser("develop")
    dev.add_argument(
        "--rebuild-makefile",
        help="Force rebuild of Makefile from template.",
        action="store_true",
    )
    dev.add_argument(
        "--xla-ref",
        help="XLA commit reference to checkout on clone",
        default=XLA_REPO_REF,
    )
    dev.add_argument(
        "--xla-dir",
        help=(
            "Set the XLA path in the Makefile. This must either be a path "
            "relative to jax_rocm_plugin or an absolute path."
        ),
        default=DEFAULT_XLA_DIR,
    )
    dev.add_argument(
        "--jax-ref",
        help="JAX commit reference to checkout on clone",
        default=TEST_JAX_REPO_REF,
    )
    dev.add_argument(
        "--kernel-jax-dir",
        help=(
            "If you want to use a local JAX directory for building the "
            "plugin kernels wheel (jax_rocm7_plugin), the path to the "
            "directory of repo. Defaults to %s" % DEFAULT_KERNELS_JAX_DIR
        ),
        default=DEFAULT_KERNELS_JAX_DIR,
    )

    doc_parser = subp.add_parser("docker")
    doc_parser.add_argument(
        "--rm",
        help="Remove the dev docker container after it exits",
        action="store_true",
    )
    return p.parse_args()


def main():
    """Run commands depending on command line input"""
    args = parse_args()
    if args.action == "docker":
        dev_docker(rm=args.rm)
    elif args.action == "develop":
        setup_development(
            xla_ref=args.xla_ref,
            xla_dir=args.xla_dir,
            test_jax_ref=args.jax_ref,
            kernels_jax_dir=args.kernel_jax_dir,
            rebuild_makefile=args.rebuild_makefile,
        )


if __name__ == "__main__":
    main()
