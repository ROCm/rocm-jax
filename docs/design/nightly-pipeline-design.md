# Nightly Pipeline Design: Dual-Pipeline Architecture

## Overview

This document describes the design rationale and architecture for the JAX ROCm nightly testing infrastructure, which consists of two complementary pipelines optimized for different testing scenarios.

## Problem Statement

The current nightly build infrastructure faces a fundamental architectural constraint: **installed JAX ROCm plugin and PJRT wheels cannot be consumed by Bazel when executing tests via RBE**.

### Root Cause

The existing nightly pipeline:
1. Builds JAX ROCm plugin and PJRT wheels
2. Installs these wheels into a Docker image
3. Attempts to run Bazel tests against the installed packages

However, Bazel with RBE requires **building from source** rather than consuming pre-installed wheel packages. The RBE build system:
- Needs access to source code and BUILD files
- Cannot reference Python packages installed via pip
- Requires all dependencies to be declared in the Bazel dependency graph

This creates an incompatibility: the wheel-based installation approach used by the current nightly cannot leverage RBE for test execution.

### Testing Requirements

We need comprehensive test coverage that includes:
- Full single-GPU,multi-GPU unit test suites (requiring RBE for reasonable build times)
- Integration validation ensuring wheels install and function correctly in end-user scenarios

## Proposed Solution

We propose a **dual-pipeline architecture** that addresses the Bazel/wheel incompatibility by using different approaches for different testing goals:

| Pipeline | Purpose | Build Method | Test Method | Test Scope |
|----------|---------|--------------|-------------|------------|
| **Nightly RBE** | Comprehensive unit testing | Build from source via RBE | Bazel test | Full multi-GPU test suite |
| **Nightly Integration** | Installation validation | Build wheels, install in Docker | Pytest | Targeted integration tests |

### Key Insight

- **Bazel tests require source builds** → Use RBE pipeline
- **User experience requires wheel installation** → Use Integration pipeline

Both are necessary: one validates code correctness, the other validates the deliverable.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Nightly Triggers                              │
│                    (schedule: cron / manual)                         │
└─────────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│       Nightly RBE Pipeline      │  │  Nightly Integration Pipeline   │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│                                 │  │                                 │
│  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐  │
│  │  Pre-built Docker Image   │  │  │  │   Build Wheels Locally    │  │
│  │  (rocm/tensorflow-build)  │  │  │  │   (in Docker container)   │  │
│  └───────────────────────────┘  │  │  └───────────────────────────┘  │
│              │                  │  │              │                  │
│              ▼                  │  │              ▼                  │
│  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐  │
│  │  Build Wheels via RBE     │  │  │  │   Build Docker Image      │  │
│  │  (run_jax_multigpu_ut.sh) │  │  │  │   (install wheels)        │  │
│  └───────────────────────────┘  │  │  └───────────────────────────┘  │
│              │                  │  │              │                  │
│              ▼                  │  │              ▼                  │
│  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐  │
│  │  Run Full Bazel Tests     │  │  │  │  Run Integration Pytests  │  │
│  │  (multi-GPU unit tests)   │  │  │  │  (smoke tests, imports)   │  │
│  └───────────────────────────┘  │  │  └───────────────────────────┘  │
│              │                  │  │              │                  │
│              ▼                  │  │              ▼                  │
│  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐  │
│  │     Upload Test Logs      │  │  │  │   Upload Test Results     │  │
│  └───────────────────────────┘  │  │  └───────────────────────────┘  │
│                                 │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

## Pipeline 1: Nightly RBE

### Purpose
Execute the comprehensive single-GPU, multi-GPU unit test suite using Bazel with Remote Build Execution for fast, distributed builds.

### Design Decisions

1. **Pre-built Docker Image**: Uses a stable, pre-built Docker image (`rocm/tensorflow-build`) rather than building images in the pipeline. This:
   - Eliminates image build time (~30-45 minutes saved)
   - Provides a consistent, tested build environment
   - Reduces pipeline complexity and failure points

2. **RBE for Compilation**: Leverages remote build execution to:
   - Distribute compilation across multiple workers
   - Utilize build caching for faster incremental builds
   - Reduce local resource requirements

3. **Single Container Execution**: Wheel building and testing occur within the same container invocation:
   - Wheels remain in container memory (no host volume mounts for wheelhouse)
   - Automatic cleanup when container exits
   - No artifacts left on runner disk

### Workflow Steps

```yaml
steps:
  - Checkout plugin and JAX repositories
  - Apply patches to JAX test repo
  - Pull pre-built Docker image
  - Configure RBE credentials
  - Run run_jax_multigpu_ut.sh (builds wheels + runs tests)
  - Upload test logs as artifacts
```

### Test Scope
- Full multi-GPU Bazel test suite (~40+ test targets)
- Tests tagged with `gpu`, `cpu` (excluding `tpu`, `config-cuda-only`)
- Includes: pmap, pjit, sharding, distributed, collective operations

## Pipeline 2: Nightly Integration

### Purpose
Validate that built wheels install correctly and basic functionality works in a clean environment.

### Design Decisions

1. **Local Wheel Building**: Builds wheels without RBE to:
   - Avoid RBE infrastructure dependencies
   - Test the non-RBE build path
   - Provide redundancy if RBE is unavailable

2. **Fresh Docker Image**: Builds a new Docker image with wheels installed:
   - Validates the installation process
   - Tests wheels in a clean environment
   - Mimics end-user installation experience

3. **Targeted Pytest Tests**: Runs a curated subset of integration tests:
   - Import validation (ensure packages load)
   - Basic device detection
   - Simple computation smoke tests
   - Signle-GPU basics
   - Multi-GPU communication basics

### Workflow Steps

```yaml
steps:
  - Checkout repositories
  - Build wheels locally (in manylinux container)
  - Build Docker image with wheels installed
  - Run integration pytest suite
  - Upload test results
```

### Test Scope
- Package import validation
- Device enumeration and detection
- Basic JAX operations on GPU
- Simple multi-device array operations

## Benefits of Dual-Pipeline Approach

### 1. Architectural Compatibility
- **RBE Pipeline**: Builds from source, compatible with Bazel RBE requirements
- **Integration Pipeline**: Tests the actual wheel installation experience

### 2. Complete Test Coverage
- **RBE Pipeline**: Runs full Bazel test suite (comprehensive unit tests)
- **Integration Pipeline**: Validates real-world installation and import scenarios

### 3. Different Failure Detection
- RBE pipeline catches: test failures, build regressions, API changes
- Integration pipeline catches: packaging errors, missing dependencies, import failures

### 4. End-to-End Validation
- RBE tests the code correctness
- Integration tests the deliverable (wheels) that users actually install

### 5. Resource Optimization
- RBE pipeline leverages distributed compute for fast builds
- Integration pipeline uses minimal resources for quick smoke tests

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Nightly RBE Pipeline | ✅ Implemented | `.github/workflows/nightly-rbe.yml` |
| Nightly Integration Pipeline | 🔲 TODO | `.github/workflows/nightly-integration.yml` |
| Multi-GPU Test Script | ✅ Exists | `jax_rocm_plugin/build/rocm/run_jax_multigpu_ut.sh` |
| Integration Test Suite | 🔲 TODO | `tests/integration/` |

## Future Considerations

1. **Test Result Database**: Upload results from both pipelines to MySQL for trend analysis
2. **Notification System**: Alert on failures with pipeline-specific context
3. **Matrix Expansion**: Add ROCm version matrix once stable
4. **Wheel Artifact Publishing**: Publish integration-tested wheels to package registry

## Appendix: Pipeline Trigger Configuration

### Development Phase (Current)
```yaml
on:
  pull_request:
    paths:
      - '.github/workflows/nightly-rbe.yml'
  workflow_dispatch:
```

### Production Phase (Future)
```yaml
on:
  schedule:
    - cron: "0 2 * * *"  # 2 AM UTC daily
  workflow_dispatch:
```
