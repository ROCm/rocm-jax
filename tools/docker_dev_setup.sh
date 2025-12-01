#!/bin/bash

PYTHON_BINARY=python3.11

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
rocm_build_number="16864"
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

install_python_v311() {
  PY_VERSION=3.11.14

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
  && tar -xzf ./Python-${PY_VERSION}.tgz \
  && rm ./Python-${PY_VERSION}.tgz \
  && cd /tmp/Python-${PY_VERSION} && ./configure --enable-optimizations --with-lto \
  && make -j"$(nproc)" && make install && ln -s -f /usr/local/bin/python3 /usr/local/bin/python \
  && cd /tmp && rm -rf /tmp/Python-${PY_VERSION} && popd

  # checking the assumption that $PYTHON_BINARY was indeed installed
  hash -r 2>/dev/null # to refresh binaries cache
  PYTHON_BINARY_PATH=$(which "$PYTHON_BINARY")
  if [[ -z "$PYTHON_BINARY_PATH" || ! -f "$PYTHON_BINARY_PATH" ]]; then
    die "Should have installed $PYTHON_BINARY, but can't find the expected executable: $PYTHON_BINARY_PATH"
  fi
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

PYTHON_BINARY_PATH=$(which "$PYTHON_BINARY")
if [[ ! -z "$PYTHON_BINARY_PATH" && -f "$PYTHON_BINARY_PATH" ]]; then
  info "$PYTHON_BINARY is found. Skipping installation."
else
  info "$PYTHON_BINARY is not installed. Proceeding with installation..."
  install_python_v311 || die "error while installing $PYTHON_BINARY"
fi

# install system deps
apt-get update
apt-get install -y \
  wget \
  curl \
  vim \
  build-essential \
  make \
  patchelf \
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
  
  # run get_rocm.py with appropriate arguments based on major version.
  if [ "$major_version" -ge 7 ]; then
    info "Running get_rocm.py with rocm_version $rocm_version, build number $rocm_build_number and build name $rocm_job_name"
    "$PYTHON_BINARY_PATH" build/tools/get_rocm.py --rocm-version "$rocm_version"  --job-name "$rocm_job_name" --build-num "$rocm_build_number"|| die "error while installing rocm"
  else
    info "Running get_rocm.py with version $rocm_version only..."
    "$PYTHON_BINARY_PATH" build/tools/get_rocm.py --rocm-version "$rocm_version" || die "error while installing rocm"
  fi
fi

VENV_PATH=$(realpath "./.venv")
[[ -d "$VENV_PATH" ]] && die "Virtual environment directory $VENV_PATH already exists. Please remove it first."

# set up a python virtualenv to install jax python packages into
info "Setting up python virtualenv at $VENV_PATH"
"$PYTHON_BINARY_PATH" -m venv "$VENV_PATH"

ACTIVATE_SCRIPT="source_venv.sh"
ACTIVATE_SCRIPT_PATH=$(realpath "./$ACTIVATE_SCRIPT")

# creating a convinience script to activate a proper venv for further use.
# This script needs to be sourced outside of this file to use proper combination of python and packages.
# Otherwise we might routinely use system python.
echo "#!/bin/bash
source \"$VENV_PATH/bin/activate\"
" > "$ACTIVATE_SCRIPT_PATH"
chmod +x "$ACTIVATE_SCRIPT_PATH"

info "Entering virtualenv"
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT_PATH"
# now we're free to use just 'python'

python -m pip install --upgrade pip

# Install Python linting tools
python -m pip install \
  black \
  pylint

# Install deps (jax and jaxlib)
python -m pip install -r \
  build/requirements.txt

info "=============================================================================="
info "ATTENTION! You MUST ALWAYS run 'source $ACTIVATE_SCRIPT' to activate the python"
info "environment at '$VENV_PATH' and to use a proper combination of python and packages."
info "(you don't need to run it for the terminal session you get after this script completion)"
info "Failing to 'source $ACTIVATE_SCRIPT' could lead to using system python instead of the $PYTHON_BINARY."
info "=============================================================================="

if [ -n "$_IS_ENTRYPOINT" ]; then
  # run CMD from docker
  if [ -n "$1" ]; then
    # shellcheck disable=SC2048
    $*
  else
    bash
  fi
else # if is run not from docker entry point, just spawn a child terminal to retain venv setting
  # otherwise a user will forget about sourcing the venv with 100% certainty
  bash
fi

# vim: sw=2 sts=2 ts=2 et
