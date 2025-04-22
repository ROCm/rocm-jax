#!/bin/bash


error() {
    echo "$*" >&2
}

die() {
    [ -n "$1" ] && error "$*"
    exit 1
}

python3 build/ci_build \
    --rocm-version 6.3.4 \
    --python-versions 3.10 \
    --compiler clang dist_wheels \
    || die "jax_rocm_plugin wheel build failed"


# copy wheels from plugin wheel build
mkdir -p wheelhouse
cp jax_rocm_plugin/wheelhouse/* wheelhouse/


python3 build/ci_build \
    --rocm-version 6.3.4 \
    build_dockers \
    || die "failed to build docker image(s) for testing"


python3 build/ci_build \
    test jax-ubu22.rocm634 \
    || die "failure during integration tests"


# vim: set ts=4 sts=4 sw=4 et:
