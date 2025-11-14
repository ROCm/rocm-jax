#!/bin/bash

set -ex

# Fail gracefully if there's no ./jax directory
if [ ! -d ./jax ]; then
    echo "ERROR: ./jax directory does not exist"
fi

# (charleshofer) This might create a dependency we don't want, but we can
# remove this once platform information is stored in the XLA project.
# Give platform information to JAX
cp -r jax_rocm_plugin/platform jax/

