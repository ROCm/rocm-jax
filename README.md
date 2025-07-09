# rocm-jax

`rocm-jax` contains sources for the ROCm plugin for JAX, as well as Dockerfiles used to build AMDs `rocm/jax` images.

# development setup

Run stack.py develop to clone jax/xla
```
python stack.py develop
```

Create a fresh ubuntu 22.04 container
```
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

## Docker environment setup

Development on the plugin is usually done in a Docker container environment
to isolate it from host systems, and allow developers to use different versions
of ROCm, CPython, and other system libraries while developing.

There are two options for setting up your Docker environment.

### Option 1 - docker setup script

Use the docker setup script in tools to set up your environment.

```
cd /rocm-jax
bash tools/docker_dev_setup.sh
```

This will do the following
  - Install system deps with apt-get
  - Install clang-18
  - Install ROCm
  - Create a python virtualenv for JAX + python packages


After this you should re-run stack.py develop to rebuild your makefile
```
python stack.py develop --rebuild-makefile
```

Now you can build the plugin
```
(cd jax_rocm_plugin && make clean dist)
```

To activate the virtual environment, run the following:
```
source .venv/bin/activate
```

To install the newly built plugin wheels, run the following command:
```
pip install jax_rocm_plugin/dist/*.whl
```


### Option 2 - manual docker setup

Install system deps
```
apt-get update
apt-get install -y \
  python3 \
  python-is-python3 \
  wget \
  curl \
  vim \
  build-essential \
  make \
  patchelf \
  python3.10-venv \
  lsb-release \
  cmake \
  yamllint \
  shellcheck
```

Install clang
```
mkdir -p /tmp/llvm-project

[[ -e /tmp/llvm-project/README.md ]] || wget -O - https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.8.tar.gz | tar -xz -C /tmp/llvm-project --strip-components 1

mkdir -p /tmp/llvm-project/build
pushd /tmp/llvm-project/build

cmake -DLLVM_ENABLE_PROJECTS='clang;lld' -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/lib/llvm-18/ ../llvm

make -j$(nproc) && make -j$(nproc) install && rm -rf /tmp/llvm-project

popd
```

Install ROCm
```
cd /rocm-jax
python build/tools/get_rocm.py --rocm-version 6.4.0
```

Create a virtualenv and activate it
```
python -m venv .venv
. .venv/bin/activate
```

Install dependencies
```
pip install -r build/requirements.txt
```

If using ROCm version >= 7, apply necessary patch for namespace change
```
patch -p1 \
    -d "$(python3 -c \"import sysconfig; print(sysconfig.get_paths()['purelib'])\")" \
    < jax_rocm_plugin/third_party/jax/namespace.patch

```

Run stack.py to refresh your local Makefile for the docker env
```
python stack.py develop --rebuild-makefile
```

Use make to build the plugin
```
(cd jax_rocm_plugin && make clean dist)
```

# Unit Testing Setup

Clone the repository
```
git clone https://github.com/ROCm/rocm-jax.git && cd rocm-jax
```

Build manylinux wheels
```
python3 build/ci_build --compiler=clang --python-versions="3.10" --rocm-version=7.0.0 --rocm-build-job="compute-rocm-dkms-no-npi-hipclang" --rocm-build-num="16306" dist_wheels
```

If you have BuildKit error:
```
sudo apt-get update
sudo apt install docker-buildx
export DOCKER_BUILDKIT=1
```

Move the created wheels to wheelhouse directory
```
mkdir -p wheelhouse && mv jax_rocm_plugin/wheelhouse/* ./wheelhouse/
```

Create docker image
```
python3 build/ci_build --rocm-version=7.0.0 --rocm-build-job="compute-rocm-dkms-no-npi-hipclang" --rocm-build-num="16306" build_dockers --filter=ubu22
```

Create container with the image created in the previous step
```
alias drun='sudo docker run --name <yourID>_rocm-jax -it --network=host  --device=/dev/infiniband --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -w /root -v /home/<yourID>/rocm-jax:/rocm-jax'
drun jax-ubu22.rocm700 OR drun <docker image id or name of the image step 5 produced>
```

To test UTs:
```
apt-get install -y vim git
cd /rocm-jax
python stack.py develop

cd jax
pip install -r build/test-requirements.txt && pip install -r build/rocm-test-requirements.txt
python ./build/rocm/run_single_gpu.py -c 2>&1 | tee 0.6.0_ut.log
```


# Nightly Builds

We build rocm-jax nightly with [a Github Actions workflow](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml).

## Docker

Nightly docker images are kept in the Github Container Registry

```
echo <MY_GITHUB_ACCESS_TOKEN> | docker login ghcr.io -u <USERNAME> --password-stdin
docker pull ghcr.io/rocm/jax-ubu24.rocm70:nightly
```

You can also find nightly images for other Ubuntu versions and ROCm version as well as older nightly images on the [packages page](https://github.com/orgs/ROCm/packages?repo_name=rocm-jax). Images get tagged with the git commit hash of the commit that the image was built from.

### Authenticating to the Container Registry

Pull access to the Github CR is done by a personal access token (classic) with the `read:packages` permission. To create one, click your profile picture in the top-right of Github, select Settings > Developer settings > Personal access tokens > Tokens (classic) and then select the option to generate a new token. Make sure you select the classic token option and git it the `read:packages` permission.

Once your token has been created, go back to the Tokens (classic) page and set your token's SSO settings to allow access to the ROCm Github organization.

Once your token has been set up to use SSO, you can log in with the `docker` command line by running,

```
echo <MY_GITHUB_ACCESS_TOKEN> | docker login ghcr.io -u <USERNAME> --password-stdin
```

## Wheels

Wheels get saved as artifacts to each run of the nightly workflow. Go to the [nightly workflow](https://github.com/ROCm/rocm-jax/actions/workflows/nightly.yml), select the run you want to get wheels from, and scroll down to the bottom of the page to find the build artifacts. Each artifact is a zip file that contains all of the wheels built for a specific ROCm version.

