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

# Default values
rocm_version="6.4.0"
rocm_build_number=""
rocm_job_name="compute-rocm-dkms-no-npi-hipclang"

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

  # if major version is >= 7, then build number and name must be provided
  if [ "$major_version" -ge 7 ]; then
    if [[ -z "$rocm_build_number" || -z "$rocm_job_name" ]]; then
      info "ERROR: For ROCm version >= 7.x, both --rocm_build_number and --rocm_job_name must be provided."
      exit 1
    fi
  fi

  info "Installing ROCm version: $rocm_version"
  
  # run get_rocm.py with appropriate arguments based on major version
  if [ "$major_version" -ge 7 ]; then
    info "Running get_rocm.py with build number $rocm_build_number and build name $rocm_job_name"
    python build/tools/get_rocm.py --rocm-version "$rocm_version"  --job-name "$rocm_job_name" --build-num "$rocm_build_number"|| die "error while installing rocm"
  else
    info "Running get_rocm.py with version only..."
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

# Apply patch for namespace change if ROCm version >= 7
if [ "$major_version" -ge 7 ]; then
  info "Applying patch for ROCm $rocm_version..."
  dist_packages=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  patch -p1 -d "$dist_packages" < jax_rocm_plugin/third_party/jax/namespace.patch
else
  info "ROCm version $rocm_version, skipping patch."
fi

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
