#!/bin/bash

FILES=$(git diff --name-only HEAD -- '*.yaml' '*.yml')
#shellcheck disable=SC2068
yamllint "$@" ${FILES[@]}

