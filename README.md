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


### Option 2 - manual docker setup

Install system deps
```
apt-get update
apt-get install -y \
  python3 \
  python-is-python3 \
  python3.10-venv \
  vim \
  git \
  build-essential \
  make \
  cmake \
  wget \
  curl
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
python build/tools/get_rocm.py --rocm-version 6.4.0
```

Create a virtualenv and activate it
```
python -m venv .venv
. .venv/bin/activate
```

Run stack.py to refresh your local Makefile for the docker env
```
python stack.py develop --rebuild-makefile
```

Use make to build the plugin
```
(cd jax_rocm_plugin && make clean dist)
```
