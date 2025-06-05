#!/bin/bash

FILES=(
  build/ci_build
  stack.py
)

mapfile -t FILES < <(find build -name "*.py")
mapfile -t FILES < <(find tools -name "*.py")

#shellcheck disable=SC2068
black -t py36 "$@" ${FILES[@]}

