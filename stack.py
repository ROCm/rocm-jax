#!/usr/bin/env python3

import argparse
import os
import subprocess


JAX_REPO_REF = "rocm-jaxlib-v0.6.0"
XLA_REPO_REF = "rocm-jaxlib-v0.6.0"


JAX_REPL_URL = "https://github.com/rocm/jax"
XLA_REPL_URL = "https://github.com/rocm/xla"


MAKE_TEMPLATE = r"""
# gfx targets for which XLA and jax custom call kernels are built for
AMDGPU_TARGETS ?= "gfx906,gfx908,gfx90a,gfx942,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201"

# customize to a single arch for local dev builds to reduce compile time
#AMDGPU_TARGETS ?= "gfx908"

.PHONY: test clean install dist

.default: dist


dist: jax_rocm_plugin jax_rocm_pjrt


jax_rocm_plugin:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-plugin \
            --rocm_path=/opt/rocm/ \
            --rocm_version=7 \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
	    --bazel_options="--override_repository=xla=../xla" \
            --verbose \
            --clang_path=%(clang_path)s


jax_rocm_pjrt:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-pjrt \
            --rocm_path=/opt/rocm/ \
            --rocm_version=7 \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
	    --bazel_options="--override_repository=xla=../xla" \
            --verbose \
            --clang_path=%(clang_path)s


clean:
	rm -rf dist


install: dist
	pip install --force-reinstall dist/*


test:
	python3 tests/test_plugin.py
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


def setup_development(jax_ref: str, xla_ref: str, rebuild_makefile: bool = False):
    # clone jax repo for jax test case source code

    if not os.path.exists("./jax"):
        cmd = ["git", "clone"]
        cmd.extend(["--branch", jax_ref])
        cmd.append(JAX_REPL_URL)
        subprocess.check_call(cmd)

    # clone xla from source for building jax_rocm_plugin
    if not os.path.exists("./xla"):
        cmd = ["git", "clone"]
        cmd.extend(["--branch", xla_ref])
        cmd.append(XLA_REPL_URL)
        subprocess.check_call(cmd)

    # create build/install/test script
    makefile_path = "./jax_rocm_plugin/Makefile"
    if rebuild_makefile or not os.path.exists(makefile_path):
        kvs = {
            "clang_path": "/usr/lib/llvm-18/bin/clang",
        }

        clang_path = find_clang()
        if clang_path:
            print("Found clang at %r" % clang_path)
            kvs["clang_path"] = clang_path
        else:
            print("No clang found. Defaulting to %r" % kvs["clang_path"])

        makefile_content = MAKE_TEMPLATE % kvs

        with open(makefile_path, "w") as mf:
            mf.write(makefile_content)


def dev_docker():
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

    cmd.append(image_name)

    p = subprocess.Popen(cmd)
    p.wait()


# build mode setup


# install jax/jaxlib from known versions
# setup build/install/test script
def setup_build():
    raise NotImplementedError


def parse_args():
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
        "--jax-ref",
        help="JAX commit reference to checkout on clone",
        default=JAX_REPO_REF,
    )

    docker = subp.add_parser("docker")

    return p.parse_args()


def main():
    args = parse_args()
    if args.action == "docker":
        dev_docker()
    elif args.action == "develop":
        setup_development(
            rebuild_makefile=args.rebuild_makefile,
            xla_ref=args.xla_ref,
            jax_ref=args.jax_ref,
        )


if __name__ == "__main__":
    main()
