# rocm-jax

## Deprecation Notice

The `rocm-jax` repository is deprecated for JAX wheel development, build, and
test workflows. **Including or any version before 0.9.2, use the release branches 
of this repository to build or test JAX**. 

| JAX Version | Repo used for building | Build jaxlib wheel?        |
|-------------|------------------------|----------------------------|
| 0.8.x       | rocm/rocm-jax          | Yes                        |
| 0.9.x       | rocm/rocm-jax          | Yes                        |
| 0.10.x      | rocm/jax               | No                         |
| latest      | rocm/jax or jax-ml/jax | (only when jaxlib changes) |


**Starting from JAX 0.10.0 Release, teams that build or test JAX 
wheels must use the ROCm JAX fork**:

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

## Building PJRT and Plugin Wheels

ROCm JAX PJRT and plugin wheels should be built from the ROCm JAX fork, not
from this infrastructure repository:

```shell
git clone https://github.com/ROCm/jax.git
cd jax
```

For a local wheel build, use the ROCm artifact build script in `ROCm/jax`.
Set `JAXCI_HERMETIC_PYTHON_VERSION` for the Python ABI you want to build, and
set `JAXCI_OUTPUT_DIR` to the directory where wheels should be written:

```shell
export JAXCI_OUTPUT_DIR="$PWD/dist"
export JAXCI_CLONE_MAIN_XLA=1

# Build one Python-specific ROCm plugin wheel.
JAXCI_HERMETIC_PYTHON_VERSION=3.12 \
  ./ci/build_rocm_artifacts.sh jax-rocm-plugin

# Build the py3-none ROCm PJRT wheel.
JAXCI_HERMETIC_PYTHON_VERSION=3.12 \
  ./ci/build_rocm_artifacts.sh jax-rocm-pjrt
```

To build plugin wheels for multiple Python versions, repeat the plugin command
with each required Python version, for example `3.11`, `3.12`, `3.13`, and
`3.14`. PJRT is a `py3-none` wheel, so it only needs to be built once for the
wheel set.

CI wheel production should use the ROCm/jax artifact workflow and scripts,
including `.github/workflows/build_rocm_artifacts.yml`,
`ci/build_rocm_artifacts.sh`, and `build/build.py`. The resulting wheels are
published by ROCm/jax and consumed from the configured CloudFront/S3 artifact
location.

The `ci/build_rocm_artifacts.sh` script is the preferred wrapper because it
sets up the ROCm build environment and invokes `build/build.py` with the ROCm
Bazel configuration. For debugging or custom automation in `ROCm/jax`, the
equivalent direct `build.py` shape is:

```shell
export JAXCI_OUTPUT_DIR="$PWD/dist"
export JAXCI_CLONE_MAIN_XLA=1

python build/build.py build \
  --wheels=jax-rocm-plugin \
  --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc" \
  --bazel_options=--config=rocm_release_wheel \
  --bazel_options=--config=rocm_rbe \
  --python_version=3.12 \
  --verbose \
  --detailed_timestamped_log \
  --output_path="$JAXCI_OUTPUT_DIR"

python build/build.py build \
  --wheels=jax-rocm-pjrt \
  --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc" \
  --bazel_options=--config=rocm_release_wheel \
  --bazel_options=--config=rocm_rbe \
  --python_version=3.12 \
  --verbose \
  --detailed_timestamped_log \
  --output_path="$JAXCI_OUTPUT_DIR"
```

Use `--wheels=jax-rocm-plugin` for Python-specific plugin wheels and
`--wheels=jax-rocm-pjrt` for the `py3-none` PJRT wheel. Change
`--python_version` for each plugin ABI that needs to be built.

## Docker Image Infrastructure

The Dockerfiles in this repository are retained for image construction. ROCm is
installed from TheRock pip wheels; the legacy apt-ROCm (7.2.0) images have been
removed.

Base and builder images:

- `docker/Dockerfile.base-therock-ubu24`
- `docker/manylinux/Dockerfile.jax-manylinux_2_28-therock`

Hermetic JAX/XLA build images:

- `docker/Dockerfile.rocm`
- `docker/Dockerfile.theRock`
- `docker/Dockerfile.rocmless`

The Dockerfiles depend on the small set of retained infrastructure inputs,
including `docker/python_requirements.txt` and `docker/patches/`.

Example image build shape:

```shell
# Build a TheRock base image.
docker build \
  -f docker/Dockerfile.base-therock-ubu24 \
  --build-arg GFX_ARCH=device-gfx942,device-gfx950 \
  --build-arg THEROCK_INDEX_URL=https://repo.amd.com/rocm/whl-multi-arch/ \
  --build-arg THEROCK_VERSION=7.14 \
  -t ghcr.io/rocm/jax-base-ubu24.therock-7.14:local \
  .

# Build a TheRock manylinux builder image.
docker build \
  -f docker/manylinux/Dockerfile.jax-manylinux_2_28-therock \
  --build-arg GFX_ARCH=device-gfx942,device-gfx950 \
  --build-arg THEROCK_INDEX_URL=https://repo.amd.com/rocm/whl-multi-arch/ \
  --build-arg THEROCK_VERSION=7.14 \
  -t ghcr.io/rocm/jax-manylinux_2_28-therock-7.14:local \
  .
```

## Running Unit Tests
First  install wheels produced in $JAXCI_OUTPUT_DIR. 
```shell
cd dist/
pip install dist/*.whl
pip install jax==<version>
```
Then confirm if GPU devices are visible
```shell
import jax
print(jax.local_devices())
```
Then run tests
```shell

# Change to jax repository root
cd jax
pip install pytest pytest-xdist pytest-html pytest-csv flatbuffers hypothesis uv
cd jax && mkdir -p dist && JAXCI_PYTHON="$(command -v python)" ./ci/run_pytest_rocm.sh
```

## Retained Automation

The `.github/workflows` directory and scripts used by those workflows are kept
so existing infrastructure can be migrated deliberately. Docker image builds
live in `.github/workflows/build-base-docker.yml` (TheRock base, dev, and
manylinux images) and `.github/workflows/build_hermetic_ubu24_docker.yml`
(hermetic JAX/XLA build images). Wheel-build and JAX-test workflows have moved
to the ROCm/jax artifact model, where
`.github/workflows/build_rocm_artifacts.yml` runs `ci/build_rocm_artifacts.sh`
and `build/build.py`. TheRock wheel builds should use the ROCm/jax TheRock
manylinux image override, such as
`ghcr.io/rocm/jax-manylinux_2_28-therock-latest:latest`.

The reporting workflows for pytest results and Llama performance remain in this
repository with their supporting upload scripts.
