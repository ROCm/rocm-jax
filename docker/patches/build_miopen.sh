#!/usr/bin/env bash
#
# Build custom MIOpen library with stack overflow kernel init fix
# This should be removed once https://github.com/ROCm/rocm-libraries/pull/4472
# lands in a ROCm release.

set -euxo pipefail

ROCM_VERSION=${ROCM_VERSION:-}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

# Skip for ROCm 7.1.1
if [ "$ROCM_VERSION" = "7.1.1" ]; then
    echo "Skipping MIOpen custom build for ROCm 7.1.1"
    exit 0
fi

# Install build dependencies
apt-get update
apt-get install -y --no-install-recommends \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libgtest-dev \
    nlohmann-json3-dev \
    libsqlite3-dev \
    libbz2-dev \
    zlib1g-dev \
    build-essential \
    cmake \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository with the fix
git clone https://github.com/ROCm/rocm-libraries.git \
    --branch fix/stack-overflow-kernel-init \
    --depth 1 \
    /tmp/rocm-libraries

# Prevent apt and ldconfig from overriding the symlinks
dpkg-divert --no-rename --remove ${ROCM_PATH}/lib/libMIOpen.so.1 || true
dpkg-divert --add --package local \
    --divert ${ROCM_PATH}/lib/libMIOpen.bak.1.0.70200 \
    --rename ${ROCM_PATH}/lib/libMIOpen.so.1.0.70200

# Build MIOpen
cd /tmp/rocm-libraries/projects/miopen/
mkdir build && cd build

# Add ROCm paths to environment
export PATH="${ROCM_PATH}/llvm/bin:${ROCM_PATH}/bin:${PATH}"
export CXX=${ROCM_PATH}/llvm/bin/amdclang++

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DMIOPEN_BACKEND=HIP \
      -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/amdclang++ \
      -DBUILD_TESTING=OFF \
      -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
      -DMIOPEN_USE_MLIR=OFF \
      -DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF \
      -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF \
      ..

make -j$(nproc)
make install

# Cleanup
cd /
rm -rf /tmp/rocm-libraries
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "MIOpen custom build completed successfully"
