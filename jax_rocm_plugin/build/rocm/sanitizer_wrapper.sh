#!/usr/bin/env bash

ASAN_RT="/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so"
TSAN_RT="/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.tsan-x86_64.so"

# Resolve suppression files from runfiles
ASAN_SUPP="$TEST_SRCDIR/jax_rocm_plugin/build/rocm/asan_ignore_list.txt"
LSAN_SUPP="$TEST_SRCDIR/jax_rocm_plugin/build/rocm/lsan_ignore_list.txt"
TSAN_SUPP="$TEST_SRCDIR/jax_rocm_plugin/build/rocm/tsan_ignore_list.txt"

ASAN_OPTS="use_sigaltstack=0:detect_leaks=0"
LSAN_OPTS="use_sigaltstack=0"
TSAN_OPTS="history_size=7:ignore_noninstrumented_modules=1"

if [[ -f "$ASAN_SUPP" ]]; then
    export LD_PRELOAD="${ASAN_RT}"
    ASAN_OPTS="suppressions=${ASAN_SUPP}:${ASAN_OPTS}"
fi
if [[ -f "$LSAN_SUPP" ]]; then
    export LD_PRELOAD="${ASAN_RT}"
    LSAN_OPTS="suppressions=${LSAN_SUPP}:${LSAN_OPTS}"
fi
if [[ -f "$TSAN_SUPP" ]]; then
    export LD_PRELOAD="${TSAN_RT}"
    TSAN_OPTS="suppressions=${TSAN_SUPP}:${TSAN_OPTS}"
fi

export ASAN_OPTIONS="${ASAN_OPTS}"
export LSAN_OPTIONS="${LSAN_OPTS}"
export TSAN_OPTIONS="${TSAN_OPTS}"

exec "$@"
