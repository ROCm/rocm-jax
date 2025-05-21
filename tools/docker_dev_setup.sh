#!/bin/bash

# shellcheck disable=SC2034
CYAN="\033[36;01m"
GREEN="\033[32;01m"
RED="\033[31;01m"
OFF="\033[0m"

info() {
  echo -e " ${GREEN}*${OFF} $*" >&2
}

error() {
  echo -e " ${RED} ERROR${OFF}: $*" >&2
}

die() {
  [ -n "$1" ] && error "$*"
  exit 1
}


install_clang_packages() {
  apt-get install -y \
    software-properties-common \
    gnupg

  # from instructions at https://apt.llvm.org/
  [[ -e llvm.sh ]] || wget https://apt.llvm.org/llvm.sh || die "error downloading LLVM install script"
  chmod +x llvm.sh || die
  bash llvm.sh 18 || die "error installing clang-18"
}

install_clang_from_source() {
  [[ -e /usr/lib/llvm-18/bin/clang ]] && return

  set -e
  
  mkdir -p /tmp/llvm-project
  
  [[ -e /tmp/llvm-project/README.md ]] || wget -O - https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.8.tar.gz | tar -xz -C /tmp/llvm-project --strip-components 1
  
  mkdir -p /tmp/llvm-project/build
  pushd /tmp/llvm-project/build
  
  cmake -DLLVM_ENABLE_PROJECTS='clang;lld' -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/lib/llvm-18/ ../llvm
  
  make -j"$(nproc)" && make -j"$(nproc)" install && rm -rf /tmp/llvm-project

  popd
  set +e
}

if [ -n "$ROCM_JAX_DIR" ]; then
  info "ROCM_JAX_DIR is ${ROCM_JAX_DIR}"
  cd "${ROCM_JAX_DIR}"
fi

# install system deps
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
  cmake || die "error installing dependencies"

# install a clang
install_clang_packages || die "error while installing clang"

# install a rocm
info "Installing ROCm"
python build/tools/get_rocm.py --rocm-version 6.4.0 || die "error while installing rocm"

# set up a python virtualenv to install jax python packages into
info "Setting up python virtualenv at .venv"
python -m venv .venv

info "Entering virtualenv"
# shellcheck disable=SC1091
. .venv/bin/activate

if [ -n "$_IS_ENTRYPOINT" ]; then
  # run CMD from docker
  if [ -n "$1" ]; then
    # shellcheck disable=SC2048
    $*
  else
    bash
  fi
fi

# vim: sw=2 sts=2 ts=2 et
