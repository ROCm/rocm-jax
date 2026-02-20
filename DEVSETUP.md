# Development Setup

A more advanced setup for developers who want an environment where they can
configure their plugin build, run tests, build their own `jax` or `jaxlib`
wheels, and iterate more quickly than possible with the full manylinux build.

If all you want is a build of the ROCm JAX plugins using your own local copy
of XLA, it will probably be much easier for you to use the `build/ci_build`
script as described in the [README's Quickbuild section](README.md#Quickbuild).

The `stack.py` script is there to help developers set up their environment. It
has two commands,

  - `docker` that will create an Ubuntu docker container with ROCm, Clang, and
    other needed dev too
  - `develop` that clones the `rocm/jax` and `rocm/xla` repositories, used for
    running unit tests and building with different branches or local changes
    to XLA, and creates `jax_rocm_plugin/Makefile` that developers can use to
    build and install the plugin and jaxlib wheels. They are encouraged to
    modify the Makefile to fit their specific build requirements.

# Setting up a Docker Development Environment

Development on the plugin is usually done in a Docker container environment
to isolate it from host systems, and allow developers to use different versions
of ROCm, CPython, and other system libraries while developing.

Run stack.py to create a container and drop into it with a shell,
```shell
python3 stack.py docker
```
It will take a few minutes to set up.

Run stack.py again to create a Makefile and clone the `rocm/xla` and
`rocm/jax` repositories,
```shell
python3 stack.py develop --rebuild-makefile
cat jax_rocm_plugin/Makefile
```

Congratulations! You're all set up to start developing the ROCm JAX plugins.
If you're new, you should take some time to familiarize yourself with the
build targets in the Makefile. Trace them down into the `build/build.py`
script and the Bazel targets that it runs.

## The Makefile

Developers are expected to modify the Makefile in `jax_rocm_plugin` to fit
their build and debugging requirements. `stack.py` gives you a handful of
targets,

  - `jax_rocm_plugin` will build the kernels plugin wheel
  - `jax_rocm_pjrt` will build the PJRT plugin wheel
  - `dist` will build both wheels and stick them in the `dist/` directory
  - `install` will install the wheels in `dist/` with `pip`
  - `test` runs the basic plugin unit tests
  - `clean` will delete all wheels in `dist/`
  - `refresh` is a shortcut for `clean dist install`
  - and a set of dedicated targets to locally build `jaxlib` when necessary:
    - `jaxlib_clean` to remove old wheels in `$jax_dir/dist/`,
    - `jaxlib` to build and
    - `jaxlib_install` to install the wheel
    - `refresh_jaxlib` is a shortcut to `jaxlib_clean jaxlib jaxlib_install`

To build and install the plugins in your virtual environment, run
```shell
source .venv/bin/activate
make clean dist install
```

If you run `pip list | grep jax`, you should now be able to see the plugin
wheels in your Python environment, along with the `jax` and `jaxlib` wheels.

You can run the tests in `jax_rocm_plugin/tests` with,
```shell
make test
```

### Building PJRT with Local XLA

By default, the Makefile is set up to build with the local copy of XLA cloned
into the top-level of the `rocm-jax` repo. You can modify the `jax_rocm_pjrt`
target to use a different XLA,
```shell
jax_rocm_pjrt:
        python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-pjrt \
            --rocm_path=/opt/rocm/ \
            --rocm_version=7 \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
            --bazel_options="--override_repository=xla=path/to/my/copy/of/xla" \
            --verbose \
            --clang_path=/usr/lib/llvm-18/bin/clang
```

Alternatively, if you have a branch in the `rocm/xla` or upstream XLA repo
that you want to use, you can fetch it with `git` from inside of the XLA clone
sitting at the top-level of the `rocm-jax` repo,
```shell
cd ../xla

# Use a branch in rocm/xla
git fetch origin
git checkout my-xla-feature-branch

# Use upstream's main branch
git remote add upstream git@github.com:openxla/xla.git
git fetch upstream
git checkout upstream/main
```

You can then rebuild and re-install your wheels with,
```shell
source .venv/bin/activate
(cd ./jax_rocm_plugin && make clean dist install)
```

### Building Kernels and their Bindings with Local JAX

The kernels wheel (`jax_rocmX_plugin`), contains several kernels along with
the C++/Python bindings that `jaxlib` uses to launch them. The code for the
kernels and their bindings lives in the upstream JAX repo, `jax-ml/jax`, and
is split between `jaxlib/gpu` and `jaxlib/rocm`. Normally, we pull this repo
in [via Bazel](https://github.com/ROCm/rocm-jax/blob/master/jax_rocm_plugin/third_party/jax/workspace.bzl#L12).

You  might want to use your local copy of `jax` in the repo's top-level to
fix a bug in the kernels or bindings. You can modify the `jax_rocm_plugin`
target to do this with a `--bazel_options` flag, similar to how we might do
this for a local copy of XLA,
```shell
jax_rocm_plugin:
        python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-plugin \
            --rocm_path=/opt/rocm/ \
            --rocm_version=7 \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
            --bazel_options="--override_repository=xla=../xla" \
            --bazel_options="--override_repository=jax=../jax" \
            --verbose \
            --clang_path=/usr/lib/llvm-18/bin/clang
```

The code in `jaxlib/gpu` tends to change frequently, so it's a good idea to
make sure that your local copy of `jax` is pointed at the JAX release that
we are currently targeting for next release (or the release you're fixing the
bug for),
```shell
cd ../jax

git remote add upstream git@github.com:jax-ml/jax.git
git fetch upstream
git checkout upstream/release/X.X.X
```

## Manual Setup

You can also set up your environment command-by-command.

Create a fresh ubuntu 22.04 container
```shell
sudo docker run -it --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --rm \
    -v ./:/rocm-jax \
    ubuntu:22.04
```

### Docker Setup Script

Use the docker setup script in tools to set up your environment.

```shell
cd /rocm-jax
./tools/docker_dev_setup.sh
```

This will do the following
  - Install system deps with apt-get
  - Install clang-18
  - Install ROCm (if needed)
  - Install Python3.11 if it's not available already
  - Create a python virtualenv for JAX + python packages
  - Activate the virtualenv for the current terminal


After this you should re-run stack.py develop to rebuild your makefile
```shell
python stack.py develop --rebuild-makefile
```

and build and install the wheels
```shell
(cd jax_rocm_plugin && make clean dist install)
```

You can fully customize your setup by modifying the 
`tools/docker_dev_setup.sh` script and `jax_rocm_plugin/Makefile` rule set,
or by doing steps manually.

NOTE: you need to run `./tools/docker_dev_setup.sh` only once. Remember to run
`source ./source_venv.sh` (or `source ./.venv/bin/activate` if you prefer)
in each new terminal session to activate a proper python virtual environment.

### Running tests with asan
You can use a regular ci script ./jax_rocm_plugin/build/rocm/ci_run_jax_ut.sh to execute u-tests under the asan config


Simple command that can be used to execute the tests is: 

```
build/rocm/ci_run_jax_ut.sh \
    --config=asan \
    --config=rocm_sgpu
```
--config=asan will enable asan build, --config=rocm_sgpu is used to execute single gpu tests, if you need multigpu tests
then use --config=rocm_mgpu. 


More configs can be found in build/rocm/jax.bazelrc and .bazelrc files. We do plan to merge
these failes into the .bazelrc file later on!


In case you need to run the test on  a specific arch which is not supported by the asan config itself you can override 
the repo_env like: 

```
build/rocm/ci_run_jax_ut.sh \
    --config=asan \
    --config=rocm_sgpu \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx945
```
this will build and execute the tests for gfx945!
