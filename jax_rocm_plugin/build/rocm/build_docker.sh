#!/bin/bash
set -e
set -x

ROCM_VERSION="${1:?Usage: $0 <rocm-version>}"
ROCM_VERSION_TAG="${ROCM_VERSION//./}"
PLUGIN_NAMESPACE="${ROCM_VERSION%%.*}"

REGISTRY="ghcr.io/rocm"
BASE_TAG="jax-base-ubu24.rocm${ROCM_VERSION_TAG}"
JAX_TAG="jax-ubu24.rocm${ROCM_VERSION_TAG}"

# Extract commit hashes from workspace files
XLA_COMMIT=$(grep -oP 'XLA_COMMIT = "\K[0-9a-f]+' jax_rocm_plugin/third_party/xla/workspace.bzl)
JAX_COMMIT=$(grep -oP 'COMMIT = "\K[0-9a-f]+' jax_rocm_plugin/third_party/jax/workspace.bzl)
ROCM_JAX_COMMIT=$(git rev-parse HEAD)

# Build base image if not available in the registry
if ! docker manifest inspect "${REGISTRY}/${BASE_TAG}:latest" >/dev/null 2>&1; then
    echo "Base image ${REGISTRY}/${BASE_TAG}:latest not found"
    exit 255
fi

# Build JAX image
docker build \
    -f docker/Dockerfile.jax-ubu24 \
    --build-arg ROCM_VERSION="${ROCM_VERSION}" \
    --build-arg ROCM_VERSION_TAG="${ROCM_VERSION_TAG}" \
    --build-arg PLUGIN_NAMESPACE="${PLUGIN_NAMESPACE}" \
    --build-arg XLA_COMMIT="${XLA_COMMIT}" \
    --build-arg JAX_COMMIT="${JAX_COMMIT}" \
    --build-arg ROCM_JAX_COMMIT="${ROCM_JAX_COMMIT}" \
    --build-context "wheels=./wheelhouse" \
    --tag "${JAX_TAG}" \
    .

echo "Built image: ${JAX_TAG}"
