# rocm-jax

[![CI](https://github.com/ROCm/rocm-jax/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/ROCm/rocm-jax/actions/workflows/ci.yml)
[![Nightly](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml/badge.svg)](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml)

`rocm-jax` contains sources for the ROCm plugin for JAX, as well as Dockerfiles used to build AMD's `rocm/jax` images.

# Nightly Builds

We build rocm-jax nightly with [a Github Actions workflow](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml).

## Docker Images

Using our Docker images is by far the simplest way to run JAX on ROCm.
Nightly Docker images are kept in the Github Container Registry

```shell
echo <MY_GITHUB_ACCESS_TOKEN> | docker login ghcr.io -u <USERNAME> --password-stdin
docker pull ghcr.io/rocm/jax-ubu24.rocm70:nightly
```

You can also find nightly images for other Ubuntu versions and ROCm version as well as older nightly images on the [packages page](https://github.com/orgs/ROCm/packages?repo_name=rocm-jax). Images get tagged with the git commit hash of the commit that the image was built from.

### Authenticating to the Container Registry

Pull access to the Github CR is done by a personal access token (classic) with the `read:packages` permission. To create one, click your profile picture in the top-right of Github, select Settings > Developer settings > Personal access tokens > Tokens (classic) and then select the option to generate a new token. Make sure you select the classic token option and git it the `read:packages` permission.

Once your token has been created, go back to the Tokens (classic) page and set your token's SSO settings to allow access to the ROCm Github organization.

Once your token has been set up to use SSO, you can log in with the `docker` command line by running,

```shell
echo <MY_GITHUB_ACCESS_TOKEN> | docker login ghcr.io -u <USERNAME> --password-stdin
```

## Wheels

Wheels get saved as artifacts to each run of the nightly workflow. Go to the [nightly workflow](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml), select the run you want to get wheels from, and scroll down to the bottom of the page to find the build artifacts. Each artifact is a zip file that contains all of the wheels built for a specific ROCm version.


# Building and Testing Yourself

More complete build instructions can be [found here](BUILDING.md).

## Quickbuild

```shell
PYTHON_VERSION=3.12
ROCM_VERSION=7.2.0

# Clear out old builds
rm -f jax_rocm_plugin/wheelhouse/*
rm -f wheelhouse/*

# Build the wheels
python3 build/ci_build \
    --python-version $PYTHON_VERSION \
    --rocm-version $ROCM_VERSION \
    dist_wheels

# Move the wheels to the wheelhouse
mkdir -p wheelhouse
cp jax_rocm_plugin/wheelhouse/* wheelhouse

# Build the Docker image for Ubuntu 24
python3 build/ci_build \
    --python-version $PYTHON_VERSION \
    --rocm-version $ROCM_VERSION \
    build_dockers \
    --filter 24

# Run basic tests
build/ci_build test jax-ubu24.rocm710:latest \
    --test-cmd "pytest jax_rocm_plugin/tests"
```

## Using a Local Copy of XLA

You can build the `jax_rocmX_pjrt` wheel with your local copy of XLA by
supplying a `--xla-source-dir` argument to the build script when you build
the wheels,
```shell
python3 build/ci_build \
    --python-version $PYTHON_VERSION \
    --rocm-version $ROCM_VERSION \
    --xla-source-dir <PATH TO MY XLA REPO> \
    dist_wheels
```

# Development Setup

For more detailed instructions on how to set up your development environment,
see the [dev setup guide](DEVSETUP.md).

## Quickstart

```shell
python3 stack.py docker
```

Once inside the container,
```shell
python3 stack.py develop --rebuild-makefile
```
