#!/usr/bin/env python3
"""Script for setting up local development environments"""

import argparse
import os
import subprocess


TEST_JAX_REPO_REF = "rocm-jaxlib-v0.8.0"
XLA_REPO_REF = "rocm-jaxlib-v0.8.0"


JAX_REPL_URL = "https://github.com/rocm/jax"
XLA_REPL_URL = "https://github.com/rocm/xla"

DEFAULT_XLA_DIR = "../xla"
DEFAULT_JAX_DIR = "../jax"


MAKE_TEMPLATE = r"""
# gfx targets for which XLA and jax custom call kernels are built for
# AMDGPU_TARGETS ?= "gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201"

# customize to a single arch for local dev builds to reduce compile time
AMDGPU_TARGETS ?= "$(shell rocminfo | grep -o -m 1 'gfx.*')"

###### auxiliary vars. Note the absence of quotes around variable values, - these vars are expected to be put into other quoted vars
# Bazel options to build repos in a certain mode.
CFG_DEBUG=--config=debug --compilation_mode=dbg --strip=never --copt=-g3 --copt=-O0 --cxxopt=-g3 --cxxopt=-O0
CFG_RELEASE_WITH_SYM=--strip=never --copt=-g3 --cxxopt=-g3

# Sets '-fdebug-prefix-map=' compiler parameter to remap source file locations from bazel's reproducible builds
# sandbox /proc/self/cwd to correct local paths. Note, external dependencies support require 'external' symlink
# in a corresponding bazel workspace root
PLUGIN_SYMBOLS=--copt=-fdebug-prefix-map=/proc/self/cwd=%(this_repo_root)s/jax_rocm_plugin --cxxopt=-fdebug-prefix-map=/proc/self/cwd=%(this_repo_root)s/jax_rocm_plugin
JAXLIB_SYMBOLS=--copt=-fdebug-prefix-map=/proc/self/cwd=%(kernels_jax_path)s --cxxopt=-fdebug-prefix-map=/proc/self/cwd=%(kernels_jax_path)s

###### --bazel_options values, must be enquoted
# Defines a value for '--bazel_options' for each of 3 build types (pjrt, plugin + jaxlib).
# By default, uses local XLA for each wheel. Redefine to whatever option is needed for your case
ALL_BAZEL_OPTIONS="--override_repository=xla=%(xla_path)s%(custom_options)s"

# PLUGIN_BAZEL_OPTIONS and JAXLIB_BAZEL_OPTIONS define pjrt&plugin specific bazel options and jaxlib specific build options.
PLUGIN_BAZEL_OPTIONS="%(plugin_bazel_options)s"
JAXLIB_BAZEL_OPTIONS="%(jaxlib_bazel_options)s"

# Use your local JAX for building the kernels in jax_rocm_plugin
# KERNELS_JAX_OVERRIDE_OPTION="--override_repository=jax=../jax"
KERNELS_JAX_OVERRIDE_OPTION="%(kernels_jax_override)s"

###


.PHONY: test clean install dist all_wheels

.default: dist


dist: jax_rocm_plugin jax_rocm_pjrt
all_wheels: clean dist jaxlib_clean jaxlib jaxlib_install install


jax_rocm_plugin:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-plugin \
            --target_cpu_features=native \
            --rocm_path=%(rocm_path)s \
            --rocm_version=%(plugin_version)s \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
            --bazel_options=${ALL_BAZEL_OPTIONS} \
            --bazel_options=${PLUGIN_BAZEL_OPTIONS} \
            --bazel_options=${KERNELS_JAX_OVERRIDE_OPTION} \
            --verbose \
            --clang_path=%(clang_path)s


jax_rocm_pjrt:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-pjrt \
            --target_cpu_features=native \
            --rocm_path=%(rocm_path)s \
            --rocm_version=%(plugin_version)s \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
            --bazel_options=${ALL_BAZEL_OPTIONS} \
            --bazel_options=${PLUGIN_BAZEL_OPTIONS} \
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
	(cd %(kernels_jax_path)s && python3 ./build/build.py build \
            --target_cpu_features=native \
            --use_clang=true \
            --clang_path=%(clang_path)s \
            --wheels=jaxlib \
            --bazel_options=${ALL_BAZEL_OPTIONS} \
            --bazel_options=${JAXLIB_BAZEL_OPTIONS} \
            --verbose \
	)


jaxlib_clean:
	rm -f %(kernels_jax_path)s/dist/*


jaxlib_install:
	pip install --force-reinstall %(kernels_jax_path)s/dist/*


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


def get_amdgpu_targets():
    """Attempt to detect AMD GPU targets using rocminfo."""
    try:
        out = subprocess.check_output(
            "rocminfo | grep -o -m 1 'gfx.*'", shell=True, stderr=subprocess.DEVNULL
        ).decode()
        target = out.strip()
        if target:
            return target
    except Exception:
        pass
    # Default targets if rocminfo fails
    return "gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201"


def _resolve_relative_paths(xla_dir: str, jax_dir: str) -> tuple[str, str, str]:
    """Transforms relative to absolute paths. This is needed to properly support
    symbolic information remapping"""
    this_repo_root = os.path.dirname(os.path.realpath(__file__))

    xla_path = (
        xla_dir
        if os.path.isabs(xla_dir)
        else os.path.abspath(f"{this_repo_root}/jax_rocm_plugin/{xla_dir}")
    )
    assert os.path.isdir(
        xla_path
    ), f"XLA path (specified as '{xla_dir}') doesn't resolve to existing directory at '{xla_path}'"

    if jax_dir:
        kernels_jax_path = (
            jax_dir
            if os.path.isabs(jax_dir)
            else os.path.abspath(f"{this_repo_root}/jax_rocm_plugin/{jax_dir}")
        )
        # pylint: disable=line-too-long
        assert os.path.isdir(
            kernels_jax_path
        ), f"XLA path (specified as '{jax_dir}') doesn't resolve to existing directory at '{kernels_jax_path}'"
    else:
        kernels_jax_path = None
    return this_repo_root, xla_path, kernels_jax_path


def _add_externals_symlink(this_repo_root: str, xla_path: str, kernels_jax_path: str):
    """Adds ./external symlink to $(bazel info output_base)/external into each path"""
    assert os.path.isabs(this_repo_root) and os.path.isabs(xla_path)
    assert not kernels_jax_path or os.path.isabs(kernels_jax_path)

    # checking 'bazel' is executable. We only support essentially bazelisk here.
    # Supporting individual bazel binaries installed by the upstream build system
    # when it can't find bazel is a TODO for the future.
    # Broad exceptions aren't a problem here
    # pylint: disable=broad-exception-caught
    try:
        v = (
            subprocess.run(
                ["bazel", "--version"],
                cwd=f"{this_repo_root}/jax_rocm_plugin",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            .stdout.decode("utf-8")
            .rstrip()
        )
        print(
            f"Bazelisk is detected (bazel=={v}), proceeding with creation of symlinks"
        )
    except Exception as e:
        print(
            "WARNING: Bazelisk is NOT detected and a wrapper for specific bazel "
            "versions isn't implemented. Symlinks to '$(bazel info output_base)/external' "
            "will not be created in each bazel workspace root, you'll have to make them manually.\n"
            f"The error was: {e}"
        )
        return

    def _link(target: str, name: str):
        if os.path.exists(name):
            print(f"Filesystem object {name} exists, skipping symlink creation.")
        else:
            os.symlink(target, name, target_is_directory=True)
            print(f"Created symlink '{name}'-->'{target}'")

    def _make_external(wrkspace: str):
        try:
            output_base = (
                subprocess.run(
                    ["bazel", "info", "output_base"],
                    cwd=wrkspace,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                .stdout.decode("utf-8")
                .rstrip()
            )
        except Exception as e:
            print(f"Failed to query 'bazel info output_base' for '{wrkspace}':{e}")
            return
        _link(f"{output_base}/external", f"{wrkspace}/external")

    _make_external(f"{this_repo_root}/jax_rocm_plugin")
    _make_external(xla_path)  # not necessary, but useful for work on XLA only
    if kernels_jax_path:
        _make_external(kernels_jax_path)


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def setup_development(
    xla_ref: str,
    xla_dir: str,
    test_jax_ref: str,
    jax_dir: str,
    rebuild_makefile: bool = False,
    fix_bazel_symbols: bool = False,
    rocm_path: str = "/opt/rocm",
    write_makefile: bool = True,
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
    if (rebuild_makefile or not os.path.exists(makefile_path) or fix_bazel_symbols) and write_makefile:
        this_repo_root, xla_path, kernels_jax_path = _resolve_relative_paths(
            xla_dir, jax_dir
        )
        if fix_bazel_symbols:
            plugin_bazel_options = "${PLUGIN_SYMBOLS}"
            jaxlib_bazel_options = "${JAXLIB_SYMBOLS}"
            custom_options = " ${CFG_RELEASE_WITH_SYM}"
            _add_externals_symlink(this_repo_root, xla_path, kernels_jax_path)
        else:  # not modifying the build unless asked
            plugin_bazel_options, jaxlib_bazel_options, custom_options = "", "", ""

        # try to detect the  namespace version from the ROCm version
        # this is expected to throw an exception if the specified ROCm path is invalid, for example
        # if there is no .info/version
        with open(
            os.path.join(rocm_path, ".info", "version"), encoding="utf-8"
        ) as versionfile:
            full_version = versionfile.readline()
        plugin_namespace_version = full_version[0]
        if plugin_namespace_version == "6":
            # note the inconsistency in numbering - ROCm 6 is "60" but ROCm 7 is "7"
            plugin_namespace_version = "60"
        elif plugin_namespace_version != "7":
            # assume that other versions will be one digit like 7
            print(f"Warning: using unexpected ROCm version {plugin_namespace_version}")

        kvs = {
            "clang_path": "/usr/lib/llvm-18/bin/clang",
            "plugin_version": plugin_namespace_version,
            "this_repo_root": this_repo_root,
            "xla_path": xla_path,
            "kernels_jax_path": kernels_jax_path,
            "plugin_bazel_options": plugin_bazel_options,
            "jaxlib_bazel_options": jaxlib_bazel_options,
            "custom_options": custom_options,
            # If the user wants to use their own JAX for building the plugin wheel
            # that contains all the jaxlib kernel code (jax_rocm7_plugin), add that
            # to the Makefile.
            "kernels_jax_override": (
                ("--override_repository=jax=%s" % kernels_jax_path)
                if kernels_jax_path
                else ""
            ),
            "rocm_path": rocm_path,
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


def build_and_install(
    xla_ref: str,
    xla_dir: str,
    test_jax_ref: str,
    jax_dir: str,
    rebuild_makefile: bool = False,
    fix_bazel_symbols: bool = False,
    rocm_path: str = "/opt/rocm",
):
    """Run develop setup, then build all wheels and install jax"""
    # Uninstall existing packages first
    print("Uninstalling existing JAX packages...")
    subprocess.run(["python3", "-m", "pip", "uninstall", "jax", "-y"], check=False)
    subprocess.run(
        [
            "python3",
            "-m",
            "pip",
            "uninstall",
            "jaxlib",
            "jax-rocm-pjrt",
            "jax-rocm-plugin",
            "jax-plugin",
            "-y",
        ],
        check=False,
    )

    # Setup repos (but don't necessarily generate Makefile)
    setup_development(
        xla_ref=xla_ref,
        xla_dir=xla_dir,
        test_jax_ref=test_jax_ref,
        jax_dir=jax_dir,
        rebuild_makefile=rebuild_makefile,
        fix_bazel_symbols=fix_bazel_symbols,
        rocm_path=rocm_path,
        write_makefile=False,  # Don't generate Makefile for direct build
    )

    this_repo_root, xla_path, jax_path = _resolve_relative_paths(xla_dir, jax_dir)
    amdgpu_targets = get_amdgpu_targets()
    clang_path = find_clang() or "/usr/lib/llvm-18/bin/clang"

    # ROCm version detection
    try:
        with open(os.path.join(rocm_path, ".info", "version"), encoding="utf-8") as f:
            full_version = f.readline().strip()
            rocm_version = full_version[0]
            if rocm_version == "6":
                rocm_version = "60"
    except Exception:
        rocm_version = "7"

    # Bazel options
    bazel_options = [f"--override_repository=xla={xla_path}"]
    if fix_bazel_symbols:
        bazel_options.extend(["--strip=never", "--copt=-g3", "--cxxopt=-g3"])

    plugin_bazel_options = list(bazel_options)
    if fix_bazel_symbols:
        plugin_bazel_options.append(
            f"--copt=-fdebug-prefix-map=/proc/self/cwd={this_repo_root}/jax_rocm_plugin"
        )
        plugin_bazel_options.append(
            f"--cxxopt=-fdebug-prefix-map=/proc/self/cwd={this_repo_root}/jax_rocm_plugin"
        )

    jax_override = f"--override_repository=jax={jax_path}" if jax_path else ""

    def run_build(cwd, wheels, extra_options=None):
        cmd = [
            "python3",
            "./build/build.py",
            "build",
            "--use_clang=true",
            f"--wheels={wheels}",
            "--target_cpu_features=native",
            f"--rocm_path={rocm_path}",
            f"--rocm_version={rocm_version}",
            f"--rocm_amdgpu_targets={amdgpu_targets}",
            f"--clang_path={clang_path}",
            "--verbose",
        ]
        opts = extra_options or bazel_options
        for opt in opts:
            cmd.append(f"--bazel_options={opt}")
        if jax_override:
            cmd.append(f"--bazel_options={jax_override}")

        print(f"Building {wheels} in {cwd}...")
        subprocess.check_call(cmd, cwd=cwd)

    # 1. Build and install jaxlib
    if jax_path:
        # Clean jaxlib dist
        subprocess.run(f"rm -f {jax_path}/dist/*", shell=True, check=True)
        run_build(jax_path, "jaxlib")
        print(f"Installing jaxlib from {jax_path}/dist...")
        subprocess.run(f"pip install --force-reinstall {jax_path}/dist/*", shell=True, check=True)

    # 2. Build plugin and pjrt
    plugin_path = os.path.join(this_repo_root, "jax_rocm_plugin")
    subprocess.run("rm -rf dist", shell=True, cwd=plugin_path, check=True)
    run_build(
        plugin_path,
        "jax-rocm-plugin",
        plugin_bazel_options,
    )
    run_build(
        plugin_path,
        "jax-rocm-pjrt",
        plugin_bazel_options,
    )

    # 3. Install plugin and pjrt
    print("Installing jax-rocm-plugin and jax-rocm-pjrt...")
    subprocess.run(
        f"pip install --force-reinstall {plugin_path}/dist/*",
        shell=True,
        check=True,
    )

    # 4. Install JAX from source
    if jax_path:
        print(f"Installing JAX from {jax_path}...")
        subprocess.check_call(["pip", "install", "."], cwd=jax_path)


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

    # Common arguments for develop and build
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--rebuild-makefile",
        help="Force rebuild of Makefile from template.",
        action="store_true",
    )
    common.add_argument(
        "--xla-ref",
        help="XLA commit reference to checkout on clone",
        default=XLA_REPO_REF,
    )
    common.add_argument(
        "--xla-dir",
        help=(
            "Set the XLA path in the Makefile. This must either be a path "
            "relative to jax_rocm_plugin or an absolute path."
        ),
        default=DEFAULT_XLA_DIR,
    )
    common.add_argument(
        "--jax-ref",
        help="JAX commit reference to checkout on clone",
        default=TEST_JAX_REPO_REF,
    )
    common.add_argument(
        "--jax-dir",
        help=(
            "If you want to use a local JAX directory for building the "
            "plugin kernels wheel (jax_rocm7_plugin), the path to the "
            "directory of repo. Defaults to %s" % DEFAULT_JAX_DIR
        ),
        default=DEFAULT_JAX_DIR,
    )
    common.add_argument(
        "--fix-bazel-symbols",
        help="When this option is enabled, the script assumes you need to build "
        "code in a release with symbolic info configuration to alleviate debugging. "
        "The script enables respective bazel options and adds 'external' symbolic "
        "links to corresponding workspaces pointing to bazel's dependencies storage.",
        action="store_true",
    )
    common.add_argument(
        "--rocm-path",
        help="Location of the ROCm to use for building Jax",
        default="/opt/rocm",
    )

    subp.add_parser("develop", parents=[common], help="Setup development environment")
    subp.add_parser(
        "build", parents=[common], help="Setup, build all wheels, and install JAX"
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
            jax_dir=args.jax_dir,
            rebuild_makefile=args.rebuild_makefile,
            fix_bazel_symbols=args.fix_bazel_symbols,
            rocm_path=args.rocm_path,
        )
    elif args.action == "build":
        build_and_install(
            xla_ref=args.xla_ref,
            xla_dir=args.xla_dir,
            test_jax_ref=args.jax_ref,
            jax_dir=args.jax_dir,
            rebuild_makefile=True,
            fix_bazel_symbols=args.fix_bazel_symbols,
            rocm_path=args.rocm_path,
        )


if __name__ == "__main__":
    main()
