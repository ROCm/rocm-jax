#!/bin/bash

FILES=$(git diff --name-only HEAD -- '*.py' build/ci_build)
#shellcheck disable=SC2068
pylint "$@" ${FILES[@]}

