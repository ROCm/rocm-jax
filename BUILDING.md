# Building

The plugin repo produces several different artifacts,
1. JAX plugin Python wheels for ROCm (`jax_rocmX_plugin` and `jax_rocmX_pjrt`)
2. Docker images with an installation of a JAX environment ready to use on ROCm
3. Unit test reports on the plugin wheels

All of these are produced by different commands to the `build/ci_build`
script. This build script does nearly all of its work inside of containers.
It requires that you have an installation of Docker and Python 3.6 or newer.

# 1. Building `jax_rocmX_plugin` and `jax_rocmX_pjrt` Wheels

Run the `build/ci_build` script,
```shell
PYTHON_VERSION="3.11,3.12"
ROCM_VERSION=6.4.1

python3 build/ci_build \
    --python-versions="$PYTHON_VERSION" \
    --rocm-version="$ROCM_VERSION" \
    dist_wheels
```

You can also build with ROCm versions that are built internally at AMD by
setting `--rocm-build-job` and `--rocm-build-num` and with your own local
copy of XLA. For example,
```shell
PYTHON_VERSION="3.11,3.12"
ROCM_VERSION=7.1.0
ROCM_BUILD_JOB=compute-rocm-dkms-no-npi-hipclang
ROCM_BUILD_NUM=16623
XLA_SOURCE=~/path/to/xla

python3 build/ci_build \
    --python-versions="$PYTHON_VERSION" \
    --rocm-version="$ROCM_VERSION" \
    --rocm-build-job="$ROCM_BUILD_JOB" \
    --rocm-build-num="$ROCM_BUILD_NUM" \
    --xla-source-dir="$XLA_SOURCE" \
    dist_wheels
```

`build/ci_build` will place your wheels inside of a wheelhouse directory.
```shell
ls jax_rocm_plugin/wheelhouse
```
If your build was successful, this should output something like,
```shell
jax_rocm7_pjrt-X.X.X-py3-none-manylinux_2_28_x86_64.whl
jax_rocm7_plugin-X.X.X-cp310-cp310-manylinux_2_28_x86_64.whl
jax_rocm7_plugin-X.X.X-cp312-cp312-manylinux_2_28_x86_64.whl
```

To install your wheels alongside `jax` and `jaxlib` in a virtual environment,
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r build/requirements.txt
pip install \
    jax_rocm_plugin/wheelhouse/jax_rocm7_pjrt-X.X.X-py3-none-manylinux_2_28_x86_64.whl
    jax_rocm_plugin/wheelhouse/jax_rocm7_plugin-X.X.X-cp310-cp310-manylinux_2_28_x86_64.whl
```

You may need to pass `--force-reinstall` to your `pip install` command if you
already have an installation of the plugin packages.

## Troubleshooting

If you have an older version of Docker on your system, you might get an error
about BuildKit not being installed or enabled. To fix, run
```shell
sudo apt-get update
sudo apt install docker-buildx
export DOCKER_BUILDKIT=1
```

If pip complains about wheels not being supported on your platform, check
the version of Python in your virtual environment and make sure that your
installing the correct plugin wheel for your Python version. 

# 2. Building Docker Images

Move the wheels created in section one to a wheelhouse directory, or download
wheels from the [nightly workflow](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml)
and place them in the wheelhouse directory. This is where Docker will look for
the wheels when building its images.
```shell
mkdir -p wheelhouse && mv jax_rocm_plugin/wheelhouse/* wheelhouse
```
Be mindful of what version of Python your Dockerfiles require. As of the
writing of this guide, we currently build images for Ubuntu 22 and Ubuntu 24,
which require Python 3.11 and Python 3.12 respectively. The kernels wheel
(`jax_rocmX_plugin`) build builds a different wheel for each Python version.

Run the build script 
```shell
ROCM_VERSION=6.4.1

python3 build/ci_build \
    --rocm-version=$ROCM_VERSION \
    build_dockers
```

Like the wheel build, you can also install versions of ROCm that were built
internally at AMD. You can also filter on the Dockerfile names in `docker/`,
and only build images from select Dockerfiles with the `--filter` option.
```shell
ROCM_VERSION=7.1.0
ROCM_BUILD_JOB=compute-rocm-dkms-no-npi-hipclang
ROCM_BUILD_NUM=16623

python3 build/ci_build \
    --rocm-version $ROCM_VERSION \
    --rocm-build-job $ROCM_BUILD_JOB \
    --rocm-build-num $ROCM_BUILD_NUM \
    build_dockers \
    --filter ubu24
```

## Troubleshooting

If your build fails with complaints about not being able to find a wheel that
satisfies `pip`'s requirements, double-check that you have a `jax_rocmX_plugin`
wheel in your wheelhouse directory that has been built for the version of
Python being installed in your Docker image.

# 3. Running Tests

JAX ROCm plugin tests are usually run in a container via the build script,
```shell
TEST_IMAGE="jax-ubu24.rocm710:latest"
python3 build/ci_build test $TEST_IMAGE --test-cmd "pytest jax_rocm_plugin/tests"
```

We keep unit tests in the `rocm/jax` repository, and you'll need to clone it
to run the regular JAX unit tests with ROCm,
```shell
git clone --depth 1 --branch rocm-jaxlib-v0.7.1 git@github.com:ROCm/jax.git
# Each release of the ROCm plugin has a corresponding branch. You can find
# more at https://github.com/ROCm/rocm-jax/branches/all?query=rocm-jaxlib

TEST_IMAGE="jax-ubu24.rocm710:latest"
python3 build/ci_build test $TEST_IMAGE --test-cmd "pytest jax/tests"
```

Once the `rocm/jax` repo is cloned, you can also use the test scripts to run
the full suite of JAX unit tests. These are handy because they run tests in
parallel on systems with multiple accelerators, and they produce reports and
logs in the `jax/logs` directory.
```shell
TEST_IMAGE="jax-ubu24.rocm710:latest"
python3 build/ci_build test $TEST_IMAGE \
    --test-cmd "python build/rocm/run_single_gpu.py -c"
```
or
```shell
TEST_IMAGE="jax-ubu24.rocm710:latest"
python3 build/ci_build test $TEST_IMAGE \
    --test-cmd "python build/rocm/run_multi_gpu.py -c"
```

## Dropping into the Container

You can also drop into the container with an interactive shell and run tests
that way. Create a container with the image from section two,
```shell
sudo docker run \
    --name <your user ID>_rocm-jax \
    -it \
    --network=host \
    --device=/dev/infiniband \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -w /root \
    -v <path to your rocm-jax>:/rocm-jax \
    jax-ubu24.rocm710:latest \
    /bin/bash
```

Once inside the container, you'll want to make sure that you have the required
Python packages for running tests,
```shell
pip install -r jax_rocm_plugin/build/test-requirements.txt

# Run the linear algebra tests, a set that exercises the core features of
# the plugin.
pytest jax/tests/linalg_test.py

# Or run the single GPU script (will take quite a while)
python jax_rocm_plugin/build/rocm/run_single_gpu.py -c
```

# How does `rocm-jax` relate to other repos?

The plugin repo pulls together code from several other repositories as part of its build.

### `jax-ml/jax`

Pulled in [via Bazel](https://github.com/ROCm/rocm-jax/blob/master/jax_rocm_plugin/third_party/jax/workspace.bzl#L12)
and is only used to [build the rocm_jaxX_plugin wheel](https://github.com/ROCm/rocm-jax/blob/master/jax_rocm_plugin/jaxlib_ext/tools/BUILD.bazel#L26).
Bazel applies a [handful of patches](https://github.com/ROCm/rocm-jax/blob/master/jax_rocm_plugin/third_party/jax/workspace.bzl#L14)
to the kernel code when it pulls jax-ml/jax. That kernel code is mostly stuff
that we share with Nvidia, changes to it from AMD are few and far in-between,
and changes almost always make their way into `jax-ml/jax` at some point, at
which we can remove the patch file.

### `rocm/xla`

Also pulled in [via Bazel](https://github.com/ROCm/rocm-jax/blob/master/jax_rocm_plugin/third_party/xla/workspace.bzl)
and is used to build the PJRT wheel (`jax_rocmX_pjrt`). XLA is a compiler
that turns operations in JAX Python code into kernels that can be run on
an AMD GPU. This happens via the PJRT interface.

### `rocm/jax`

Only used for test cases and staging PRs from AMD developers into
jax-ml/jax. Mostly, this is storing test cases that we skip because we have
yet to fix the underlying bug, and we'll eventually unskip that test case
when it gets fixed.


