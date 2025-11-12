#!/bin/bash

error() {
    echo "$*" >&2
}

die() {
    [ -n "$1" ] && error "$*"
    exit 1
}

python3 build/ci_build \
    --rocm-version 7.2.0 \
    --python-versions 3.12 \
    --compiler clang dist_wheels \
    || die "jax_rocm_plugin wheel build failed"


# copy wheels from plugin wheel build
mkdir -p wheelhouse
cp jax_rocm_plugin/wheelhouse/* wheelhouse/


python3 build/ci_build \
    --rocm-version 7.2.0 \
    build_dockers \
    || die "failed to build docker image(s) for testing"


python3 build/ci_build \
    test jax-ubu22.rocm7100 \
    || die "failure during integration tests"


# vim: set ts=4 sts=4 sw=4 et:
