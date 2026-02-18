#!/bin/bash
set -e
set -x

ROCM_VERSION="${1:?Usage: $0 <rocm-version>}"
ROCM_VERSION_TAG="${ROCM_VERSION//./}"
PLUGIN_NAMESPACE="${ROCM_VERSION%%.*}"

REGISTRY="ghcr.io/rocm"
BASE_TAG="jax-base-ubu24.rocm${ROCM_VERSION_TAG}"
JAX_TAG="jax-ubu24.rocm${ROCM_VERSION_TAG}"

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
    --build-context "wheels=./wheelhouse" \
    --tag "${JAX_TAG}" \
    .

echo "Built image: ${JAX_TAG}"
