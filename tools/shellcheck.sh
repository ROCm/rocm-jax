#!/bin/bash

FILES=$(git diff --name-only HEAD -- '*.sh')
#shellcheck disable=SC2068
shellcheck "$@" ${FILES[@]}

