#!/usr/bin/env bash
set -euo pipefail
mkdir -p bin
echo "Compiling Lab6 generator..."
gcc -std=c11 -O2 ../../lab-6/main.c -o bin/icg
echo "Built bin/icg"