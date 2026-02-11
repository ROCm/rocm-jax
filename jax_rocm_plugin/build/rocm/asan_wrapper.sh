#!/usr/bin/env bash

ASAN_RT="/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so"

# Only set LD_PRELOAD for the test binary, not for Bazel's wrappers
export LD_PRELOAD="${ASAN_RT}"

# Resolve suppression files from runfiles
ASAN_SUPP="$TEST_SRCDIR/xla/build_tools/rocm/asan_ignore_list.txt"
LSAN_SUPP="$TEST_SRCDIR/xla/build_tools/rocm/lsan_ignore_list.txt"

ASAN_OPTS="use_sigaltstack=0:detect_leaks=0"
LSAN_OPTS="use_sigaltstack=0"

ASAN_OPTS="suppressions=${ASAN_SUPP}:${ASAN_OPTS}"
LSAN_OPTS="suppressions=${LSAN_SUPP}:${LSAN_OPTS}"

export ASAN_OPTIONS="${ASAN_OPTS}"
export LSAN_OPTIONS="${LSAN_OPTS}"

exec "$@"
