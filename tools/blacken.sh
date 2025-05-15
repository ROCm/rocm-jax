#!/bin/bash

FILES=(
  build/ci_build
  stack.py
)

FILES+=($(find build -name "*.py"))
FILES+=($(find tools -name "*.py"))

black -t py36 $* ${FILES[@]}
