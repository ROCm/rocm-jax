# Design Proposal: Simplify Build System by Removing `build.py` Wrapper

## Executive Summary

The current `build.py` script adds unnecessary complexity to the ROCm JAX build system. Most of its functionality can be replaced with Bazel configuration files and direct Bazel commands, significantly simplifying the build process while maintaining all necessary functionality.

## Current State

### Architecture

The current build flow is:
1. `build.py` → constructs complex Bazel commands with many flags
2. `bazel run` → builds artifacts
3. Wheel builder scripts → post-process and package wheels

### What `build.py` Currently Does

The `jax_rocm_plugin/build/build.py` script (746 lines) performs:

- **Configuration Management & Auto-Detection**
  - Detects Bazel (downloads if missing)
  - Finds compiler paths (Clang/GCC) and versions
  - Detects Python versions
  - Detects ROCm versions and paths
  - Validates configurations

- **Flag Construction**
  - Dynamically constructs Bazel commands with many flags
  - Generates `.jax_configure.bazelrc` file
  - Handles platform-specific differences

- **Argument Parsing & Validation**
  - Complex CLI with many options
  - Validates arguments before Bazel execution
  - Loops through multiple wheels

- **Git Hash Detection**
  - Detects git hashes for versioning

## Problem Statement

### Issues with Current Approach

1. **Over-Engineering**: Most configuration can be handled by Bazel's native `.bazelrc` files with configs
2. **Unnecessary Abstraction**: Adds a layer that doesn't provide significant value
3. **Maintenance Burden**: 746 lines of Python code to maintain
4. **Redundant Validation**: Bazel already validates arguments
5. **Unnecessary Loops**: Only 2 wheels need to be built (`jax-rocm-plugin` and `jax-rocm-pjrt`)

### What's Actually Necessary

The **only** functionality that cannot be replaced by Bazel is:

1. **Python Wheel Assembly**
   - Copying files from Bazel runfiles into wheel structure
   - Generating `setup.py`/`setup.cfg` dynamically
   - Calling `python -m build -n -w` to create `.whl` files
   - Implemented in wheel builder scripts

2. **Commit Info Generation**
   - Writing commit hashes into files
   - Can be done at shell level with `$(git rev-parse HEAD)`

### What Can Be Eliminated

**Post-Build Rpath Fixing** (`patchelf`) - Currently Required but Can Be Removed:
- Currently modifies shared libraries after Bazel builds them
- The comment in code says: "setting rpath using bazel requires the rpath to be valid during the build which won't be correct until we make changes to the xla/tsl/jax plugin build"
- **Solution**: Pass rpath configuration via `repo_env` to XLA during build
- XLA can be configured to set additional rpaths (e.g., `$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib:/opt/rocm/lib`) during linking
- This eliminates the need for `patchelf` and the post-processing step entirely

## Proposed Solution

### Simplified Architecture

Replace `build.py` with:

1. **Enhanced `.bazelrc` Configuration**
   - Hardcode compiler paths (we know our environment)
   - Define configs for all build variants
   - Remove need for dynamic flag construction

2. **Direct Bazel Commands**
   - Simple, straightforward commands
   - No wrapper script needed
   - Can be in Makefile, shell script, or CI directly

3. **Simplify Wheel Builder Scripts**
   - Still needed for Python wheel assembly
   - Can remove `patchelf` step if rpaths are set during build via `repo_env`
   - Already invoked via `bazel run`

### Implementation

#### Step 1: Enhance `.bazelrc`

Add configs to `jax_rocm_plugin/.bazelrc`:

```bazelrc
# ROCm build configs
build:rocm_build --config=rocm_base
build:rocm_build --config=rocm
build:rocm_build --action_env=CLANG_COMPILER_PATH="/usr/lib/llvm-18/bin/clang"
build:rocm_build --action_env=ROCM_PATH="/opt/rocm"
build:rocm_build --action_env=TF_ROCM_AMDGPU_TARGETS="gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201"

# Rpath configuration for wheel installation
# Pass additional rpaths to XLA via repo_env so they're set during build
# This eliminates the need for post-build patchelf
build:rocm_build --repo_env=ROCM_WHEEL_RPATH="$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib:/opt/rocm/lib"

# Python version configs
build:python311 --repo_env=HERMETIC_PYTHON_VERSION=3.11
build:python312 --repo_env=HERMETIC_PYTHON_VERSION=3.12

# CPU feature configs (already exist)
build:avx_posix --copt=-mavx
build:native_arch_posix --copt=-march=native
```

**Note**: The `ROCM_WHEEL_RPATH` environment variable would need to be consumed by XLA's build system to add these rpaths during linking. This requires a small change in XLA to read this env var and append it to linker flags.

#### Step 2: Replace `build.py` with Simple Commands

**Option A: Makefile**

```makefile
GIT_HASH := $(shell git rev-parse HEAD)
ROCM_VERSION := 7
CPU := x86_64
OUTPUT_PATH := dist

.PHONY: jax-rocm-plugin jax-rocm-pjrt wheels

jax-rocm-plugin:
	bazel run //jaxlib_ext/tools:build_gpu_kernels_wheel \
		--config=rocm_build \
		--config=python311 \
		--config=avx_posix \
		-- \
		--output_path=$(OUTPUT_PATH) \
		--cpu=$(CPU) \
		--platform_version=$(ROCM_VERSION) \
		--enable-rocm=True \
		--rocm_jax_git_hash=$(GIT_HASH) \
		--xla-commit=$$(cd ../xla && git rev-parse HEAD) \
		--jax-commit=$$(cd ../jax && git rev-parse HEAD)

jax-rocm-pjrt:
	bazel run //pjrt/tools:build_gpu_plugin_wheel \
		--config=rocm_build \
		--config=python311 \
		--config=avx_posix \
		-- \
		--output_path=$(OUTPUT_PATH) \
		--cpu=$(CPU) \
		--platform_version=$(ROCM_VERSION) \
		--enable-rocm=True \
		--rocm_jax_git_hash=$(GIT_HASH) \
		--xla-commit=$$(cd ../xla && git rev-parse HEAD) \
		--jax-commit=$$(cd ../jax && git rev-parse HEAD)

wheels: jax-rocm-plugin jax-rocm-pjrt
```

**Option B: Simple Shell Script**

```bash
#!/bin/bash
set -e

GIT_HASH=$(git rev-parse HEAD)
ROCM_VERSION=7
CPU=x86_64
OUTPUT_PATH=dist

# Build plugin wheel
bazel run //jaxlib_ext/tools:build_gpu_kernels_wheel \
	--config=rocm_build \
	--config=python311 \
	--config=avx_posix \
	-- \
	--output_path="$OUTPUT_PATH" \
	--cpu="$CPU" \
	--platform_version="$ROCM_VERSION" \
	--enable-rocm=True \
	--rocm_jax_git_hash="$GIT_HASH" \
	--xla-commit="$(cd ../xla && git rev-parse HEAD)" \
	--jax-commit="$(cd ../jax && git rev-parse HEAD)"

# Build PJRT wheel
bazel run //pjrt/tools:build_gpu_plugin_wheel \
	--config=rocm_build \
	--config=python311 \
	--config=avx_posix \
	-- \
	--output_path="$OUTPUT_PATH" \
	--cpu="$CPU" \
	--platform_version="$ROCM_VERSION" \
	--enable-rocm=True \
	--rocm_jax_git_hash="$GIT_HASH" \
	--xla-commit="$(cd ../xla && git rev-parse HEAD)" \
	--jax-commit="$(cd ../jax && git rev-parse HEAD)"
```

**Option C: Direct CI Commands**

Just use the Bazel commands directly in CI workflows without any wrapper.

## Benefits

### Advantages

1. **Simpler Codebase**: Remove ~746 lines of Python code
2. **Easier to Understand**: Direct Bazel commands are self-documenting
3. **Less Maintenance**: Fewer moving parts, less code to maintain
4. **Better Performance**: No Python overhead for simple command construction
5. **More Flexible**: Users can easily customize builds by modifying bazelrc or command flags
6. **Standard Practice**: Uses Bazel's native configuration system as intended

### What We Keep

- All functionality remains intact
- Wheel builder scripts handle Python wheel assembly (can be simplified by removing patchelf)
- Build outputs are identical (or better, with rpaths set correctly from the start)
- CI/CD integration remains straightforward


### Additional Benefits: Eliminating Patchelf

If rpaths are configured via `repo_env`:
- **Remove `patchelf` dependency**: No longer needed in build environment
- **Simpler wheel builders**: Remove ~40 lines of patchelf code from each wheel builder script
- **Better build correctness**: Rpaths set correctly from the start, not patched afterward
- **Faster builds**: No post-processing step needed

## Migration Plan

### Phase 1: Preparation
1. **Add rpath support to XLA** (if not already present):
   - Modify XLA build to read `ROCM_WHEEL_RPATH` env var
   - Append rpaths to linker flags during shared library linking
   - Test that rpaths are correctly set in built artifacts
2. Enhance `.bazelrc` with all necessary configs including `ROCM_WHEEL_RPATH`
3. Document the new approach
4. Test builds with new commands

### Phase 2: Implementation
1. Remove patchelf code from wheel builder scripts
2. Remove `patchelf` dependency from build requirements
3. Create Makefile/shell script alternative
4. Update CI workflows to use new commands
5. Update documentation (DEVSETUP.md, README.md)

### Phase 3: Deprecation
1. Mark `build.py` as deprecated
2. Add deprecation warnings
3. Keep for backward compatibility during transition period

### Phase 4: Removal
1. Remove `build.py` after transition period
2. Clean up related utilities if no longer needed

## Risks and Mitigation


### Potential Risks

1. **Breaking Existing Workflows**: Users/CI may depend on `build.py`
   - **Mitigation**: Keep during transition period, provide migration guide

2. **Loss of Some Convenience Features**: Some edge cases handled by `build.py`
   - **Mitigation**: Document how to handle edge cases with Bazel configs

3. **Multiple Build Environments**: Different environments may need different configs
   - **Mitigation**: Use `.bazelrc.user` for local overrides, CI-specific bazelrc files

## Alternatives Considered

### Alternative 1: Keep `build.py` but Simplify It
- **Pros**: Maintains existing interface
- **Cons**: Still adds unnecessary abstraction layer

### Alternative 2: Use Bazel Macros
- **Pros**: More Bazel-native approach
- **Cons**: Still requires learning curve, less flexible than direct commands

### Alternative 3: Keep Current System
- **Pros**: No migration needed
- **Cons**: Maintains unnecessary complexity


## Conclusion

The proposed simplification removes unnecessary abstraction while maintaining all essential functionality. The build system becomes more maintainable, easier to understand, and follows Bazel best practices. 

By configuring rpaths via `repo_env` to XLA, we can eliminate the `patchelf` post-processing step entirely, further simplifying the wheel builder scripts. This approach sets rpaths correctly during the build rather than patching them afterward, which is more correct and maintainable.

The wheel builder scripts are still needed for Python wheel assembly, but can be significantly simplified by removing the patchelf step.

## References

- Current `build.py`: `jax_rocm_plugin/build/build.py`
- Wheel builders: 
  - `jax_rocm_plugin/jaxlib_ext/tools/build_gpu_kernels_wheel.py` (lines 174-212 contain patchelf code that can be removed)
  - `jax_rocm_plugin/pjrt/tools/build_gpu_plugin_wheel.py` (lines 167-194 contain patchelf code that can be removed)
- Bazel config: `jax_rocm_plugin/.bazelrc`
- Generated config: `.jax_configure.bazelrc` (can be removed)

## Implementation Notes

### Rpath Configuration via repo_env

The current rpath being set by patchelf is:
```
$ORIGIN/../rocm/lib:$ORIGIN/../../rocm/lib:/opt/rocm/lib
```

This needs to be passed to XLA via `--repo_env=ROCM_WHEEL_RPATH="..."` so XLA can add these rpaths during linking. The XLA build system would need to:

1. Read the `ROCM_WHEEL_RPATH` environment variable
2. Append these rpaths to the linker flags (e.g., `-Wl,-rpath,$ORIGIN/../rocm/lib:...`)
3. Ensure both build-time rpaths (for linking) and runtime rpaths (for wheel installation) are set


This eliminates the need for the patchelf post-processing step entirely.

