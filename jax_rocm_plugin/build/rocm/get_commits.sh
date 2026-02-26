#!/bin/bash
# Extracts commit hashes from workspace files and prints them as
# shell-evaluable assignments. Usage:
#   eval "$(bash jax_rocm_plugin/build/rocm/get_commits.sh)"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THIRD_PARTY="${SCRIPT_DIR}/../../third_party"

XLA_COMMIT=$(grep -oP 'XLA_COMMIT = "\K[0-9a-f]+' "${THIRD_PARTY}/xla/workspace.bzl")
JAX_COMMIT=$(grep -oP 'COMMIT = "\K[0-9a-f]+' "${THIRD_PARTY}/jax/workspace.bzl")
ROCM_JAX_COMMIT=$(git rev-parse HEAD)

echo "XLA_COMMIT=${XLA_COMMIT}"
echo "JAX_COMMIT=${JAX_COMMIT}"
echo "ROCM_JAX_COMMIT=${ROCM_JAX_COMMIT}"
