# rocm-jax

## Deprecation Notice

The `rocm-jax` repository is deprecated for JAX wheel development, build, and
test workflows. Teams that build or test JAX wheels must use the ROCm JAX fork:

```shell
git clone https://github.com/ROCm/jax.git
```

Use the build and test scripts from that repository, including `build/build.py`
and the ROCm artifact workflow. The legacy `stack.py` entrypoint in this
repository now exits with a deprecation notice. The retained `build/ci_build`
script remains available only for Docker image infrastructure actions; wheel
build and test actions exit with guidance to ROCm/jax. The
`jax_rocm_plugin/build/build.py` compatibility entrypoint delegates to
ROCm/jax `build/build.py`.

## What Remains Here

The default branch for this repository is `rocm-jax-infra`. It keeps the
Dockerfiles and infrastructure files needed to build ROCm JAX images.

This branch is not the source of truth for:

- ROCm JAX plugin or PJRT source development.
- Building JAX, `jaxlib`, ROCm plugin, or ROCm PJRT wheels.
- Running JAX unit tests for wheel validation.

Those workflows belong in `https://github.com/ROCm/jax`.

## Usage by Team

Developers and CI jobs that build or test JAX wheels should clone
`https://github.com/ROCm/jax` and follow the build and test documentation in
that repository.

QA and infrastructure jobs that build ROCm JAX images should continue using the
Dockerfiles in this `rocm-jax-infra` branch. These image builds consume wheels
produced by the ROCm JAX fork.

Image users should consume the published ROCm JAX images from the configured
container registry for their environment.

## Docker Image Infrastructure

The Dockerfiles in this repository are retained for image construction:

- `docker/Dockerfile.base-ubu24`
- `docker/Dockerfile.base-therock-ubu24`
- `docker/Dockerfile.jax-ubu24`
- `docker/manylinux/Dockerfile.jax-manylinux_2_28-rocm`
- `docker/manylinux/Dockerfile.jax-manylinux_2_28-therock`

The Dockerfiles depend on the small set of retained infrastructure inputs,
including `tools/get_rocm.py`, `build/requirements.txt`,
`docker/manylinux/clang.cfg`, `docker/patches/rocr-intercept-queue-fix.patch`,
and a local `wheelhouse/` containing ROCm JAX wheels.

Example image build shape:

```shell
# Build or obtain wheels from https://github.com/ROCm/jax first.
mkdir -p wheelhouse
cp /path/to/jax/wheels/*.whl wheelhouse/

# Build a base image.
docker build \
  -f docker/Dockerfile.base-ubu24 \
  --build-arg ROCM_VERSION=7.2.0 \
  --build-arg ROCM_VERSION_TAG=720 \
  -t ghcr.io/rocm/jax-base-ubu24.rocm720:local \
  .

# Build a JAX image from those wheels.
docker build \
  -f docker/Dockerfile.jax-ubu24 \
  --build-arg ROCM_VERSION_TAG=720 \
  --build-arg BASE_IMAGE_TAG=local \
  --build-arg ROCM_VERSION=7.2.0 \
  --build-arg JAX_VERSION=<jax-version> \
  --build-arg XLA_COMMIT=<xla-commit> \
  --build-arg JAX_COMMIT=<jax-commit> \
  --build-arg ROCM_JAX_COMMIT=<rocm-jax-commit> \
  --build-arg PLUGIN_NAMESPACE=7 \
  -t ghcr.io/rocm/jax-ubu24.rocm720:local \
  .
```

## Retained Automation

The `.github/workflows` directory and scripts used by those workflows are kept
so existing infrastructure can be migrated deliberately. Existing Docker image
workflows continue to use this repository and `build/ci_build` for Docker
image construction. Wheel-build and JAX-test workflows should move to the
ROCm/jax artifact model, where `.github/workflows/build_rocm_artifacts.yml`
runs `ci/build_rocm_artifacts.sh` and `build/build.py`. TheRock wheel builds
should use the ROCm/jax TheRock manylinux image override, such as
`ghcr.io/rocm/jax-manylinux_2_28-therock-latest:latest`.

The reporting workflows for pytest results and Llama performance remain in this
repository with their supporting upload scripts.
