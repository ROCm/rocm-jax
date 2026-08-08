#!/usr/bin/env bash
#
# Rebuild CLR (libamdhip64) with a corrected fat-binary bound and splice it over
# the copy shipped in the TheRock ROCm wheels.
#
# rocm-systems PR #9019 derives the length of the image passed to
# hipModuleLoadData() from the single /proc/self/maps entry that contains the
# image's first byte, and rejects the load when the code object does not fit
# inside it. A malloc'd buffer can span several contiguous but unmerged [heap]
# VMAs, so valid and fully readable code objects are rejected with
# hipErrorInvalidImage. clr-coalesce-readable-vmas.patch coalesces forward over
# adjacent readable mappings instead.
#
# Remove this script, the patch and the Dockerfile step once the fix reaches a
# TheRock nightly.

set -euxo pipefail

CLR_SOURCE_REF="${CLR_SOURCE_REF:?set CLR_SOURCE_REF to a rocm-systems branch or commit}"
CLR_PATCH="${CLR_PATCH:-/tmp/clr-fix.patch}"
SRC=/tmp/rocm-systems

apt-get update
apt-get install -y --no-install-recommends \
    git \
    cmake \
    pkg-config \
    libelf-dev \
    libdrm-dev \
    libnuma-dev \
    mesa-common-dev \
    libgl1-mesa-dev \
    libglx-dev \
    libgl-dev
python3 -m pip install --break-system-packages CppHeaderParser

SDK="$(rocm-sdk path --root)"

# Sparse checkout: CLR needs the HIP headers from projects/hip alongside it.
# Fetched by explicit ref rather than `clone --branch` so that CLR_SOURCE_REF may
# be a commit SHA -- a full 40-character one, since GitHub will not serve an
# abbreviated SHA to `git fetch`. Pinning matters here: CLR's device sources
# track the HSA headers of their own release line, so a ref from a newer line
# than the SDK installed above fails to compile on identifiers the SDK's headers
# do not have yet.
mkdir -p "$SRC"
cd "$SRC"
git init -q
git remote add origin https://github.com/ROCm/rocm-systems.git
git fetch --depth=1 --filter=blob:none origin "${CLR_SOURCE_REF}"
git sparse-checkout init --cone
git sparse-checkout set projects/clr projects/hip
git checkout -q FETCH_HEAD
echo "CLR source: $(git rev-parse HEAD) (${CLR_SOURCE_REF})"

# No --3way on purpose. When the patch stops applying the build must fail loudly
# rather than silently shipping a stock CLR, since the most likely cause is that
# the upstream fix landed and this hotpatch should be dropped.
git apply "$CLR_PATCH"

# ROCM_KPACK_ENABLED=ON and the default __HIP_ENABLE_PCH mirror TheRock's own
# configuration (TheRock/core/CMakeLists.txt). Without them a different
# fat-binary path is selected, yielding a ~6.5 MB library that segfaults under
# multi-process load instead of the expected ~28 MB one. HIPCC_BIN_DIR is
# deliberately empty, as TheRock leaves it.
mkdir -p projects/clr/build
cd projects/clr/build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$SDK/bin/amdclang" \
      -DCMAKE_CXX_COMPILER="$SDK/bin/amdclang++" \
      -DHIP_PLATFORM=amd \
      -DCLR_BUILD_HIP=ON \
      -DCLR_BUILD_OCL=OFF \
      -DROCM_KPACK_ENABLED=ON \
      -DUSE_NEW_HOSTCALL_IMPL=OFF \
      -DHIP_COMMON_DIR="$SRC/projects/hip" \
      -DHIPCC_BIN_DIR= \
      -DROCM_PATH="$SDK" \
      -DCMAKE_PREFIX_PATH="$SDK;$SDK/llvm" \
      -Damd_comgr_DIR="$SDK/lib/cmake/amd_comgr" \
      -Drocm-kpack_DIR="$SDK/lib/cmake/rocm-kpack" \
      -DLLVM_DIR="$SDK/llvm/lib/cmake/llvm" \
      ..
make -j"$(nproc)"

NEW="$(ls hipamd/lib/libamdhip64.so.7.*-*)"
NEW_SIZE="$(stat -c%s "$NEW")"
echo "Built $NEW ($NEW_SIZE bytes)"
if [ "$NEW_SIZE" -lt 20000000 ]; then
    echo "ERROR: $NEW is ${NEW_SIZE} bytes; expected >20 MB." >&2
    echo "A short library means the kpack/PCH flags did not take effect." >&2
    exit 1
fi

# Replace every copy, not just the one rocm-sdk points at. The wheels ship
# libamdhip64 in two packages: `rocm-sdk path --root` returns _rocm_sdk_devel,
# but these images export no ROCm env, so JAX resolves ROCm through its
# $ORIGIN-relative RUNPATHs and maps the _rocm_sdk_core copy. Some of the devel
# paths are hardlinks to the core copy and some are separate inodes, so patching
# only one of them has no effect on a normally-run container.
SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
mapfile -t TARGETS < <(
    find "$SP/_rocm_sdk_core/lib" "$SP/_rocm_sdk_devel/lib" \
        -name 'libamdhip64.so*' -type f | sort
)
if [ "${#TARGETS[@]}" -lt 4 ]; then
    printf 'ERROR: found %d libamdhip64 copies, expected at least 4:\n' \
        "${#TARGETS[@]}" >&2
    printf '  %s\n' "${TARGETS[@]}" >&2
    exit 1
fi

# Write through the existing inodes so hardlinks and symlinks stay intact.
for f in "${TARGETS[@]}"; do
    cat "$NEW" > "$f"
done

WANT="$(sha256sum "$NEW" | cut -d' ' -f1)"
for f in "${TARGETS[@]}"; do
    got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" != "$WANT" ]; then
        echo "ERROR: $f was not replaced ($got != $WANT)" >&2
        exit 1
    fi
    echo "patched $f"
done

cd /
rm -rf "$SRC"
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "CLR hotpatch applied to ${#TARGETS[@]} copies of libamdhip64"
