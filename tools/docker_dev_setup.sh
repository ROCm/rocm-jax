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

# Set timezone up front (pick your zone)
ln -fs /usr/share/zoneinfo/UTC /etc/localtime
# or: export TZ=UTC

# Install tzdata without prompting
TZ=UTC DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
dpkg-reconfigure --frontend noninteractive tzdata

# Default values
rocm_version="7.2.0"

# Parse named command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rocm_version) rocm_version="$2"; shift ;;
        --rocm_build_number) rocm_build_number="$2"; shift ;;
        --rocm_job_name) rocm_job_name="$2"; shift ;;
        *) error "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

install_python_v311() {
  #install python3.11
PY_VERSION=3.11.13

# System packages needed to compile Python and installed bzip2, sqlite3 packages
apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential            \
    wget curl ca-certificates  \
    libssl-dev zlib1g-dev      \
    libreadline-dev libffi-dev \
    sqlite3 libsqlite3-dev     \
    libbz2-dev liblzma-dev     \
    libncursesw5-dev xz-utils  \
    tk-dev uuid-dev            \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and unpack python-v3.11
 pushd . \
 && mkdir -p /tmp && cd /tmp && wget https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz \
 && tar -xzf Python-${PY_VERSION}.tgz \
 && rm Python-${PY_VERSION}.tgz \
 && cd /tmp/Python-${PY_VERSION} && ./configure --enable-optimizations --with-lto \
 && make -j"$(nproc)" && make install && ln -s /usr/local/bin/python3 /usr/local/bin/python \
 && popd
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

install_python_v311 || die "error while installing python3.11"

# install system deps
apt-get update
apt-get install -y \
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
  shellcheck \
  git || die "error installing dependencies"
# install a clang
install_clang_packages || die "error while installing clang"

# install ROCm (if needed)
# extract the major version
major_version=$(echo "$rocm_version" | cut -d. -f1)

# check if ROCm is installed using rocminfo or fallback to checking /opt/rocm
if command -v rocminfo &> /dev/null; then
  info "ROCm is already installed (found via rocminfo). Skipping installation."
elif [[ -d "/opt/rocm" ]]; then
  info "ROCm directory found at /opt/rocm. Assuming ROCm is installed. Skipping installation."
else
  info "ROCm is not installed. Proceeding with installation..."

  # run get_rocm.py with appropriate arguments based on major version
  if [[ -n "$rocm_build_number" && -n "$rocm_job_name" ]]; then
    info "Running get_rocm.py with rocm_version $rocm_version, build number $rocm_build_number and build name $rocm_job_name"
    python build/tools/get_rocm.py --rocm-version "$rocm_version"  --job-name "$rocm_job_name" --build-num "$rocm_build_number"|| die "error while installing rocm"
  else
    info "Running get_rocm.py with version $rocm_version only..."
    python build/tools/get_rocm.py --rocm-version "$rocm_version" || die "error while installing rocm"
  fi
fi

# set up a python virtualenv to install jax python packages into
info "Setting up python virtualenv at .venv"
python -m venv .venv

info "Entering virtualenv"
# shellcheck disable=SC1091
. .venv/bin/activate

# Install Python linting tools
python -m pip install \
  black \
  pylint

# Install deps (jax and jaxlib)
python -m pip install -r \
  build/requirements.txt

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
