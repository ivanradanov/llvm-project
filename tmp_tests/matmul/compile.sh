#!/bin/bash

set -x
set -e

CCX="../../build-clang/bin/clang++"
#OPT="../../build/bin/opt"
#OPT="../../build/bin/llvm-link"

$CCX -c matmul.cpp -O3 -flto

../../build/bin/opt matmul.o -passes=cpucuda -o matmul_cpu.o

$CCX -c main.cpp -O3 -flto

../../build/bin/llvm-link main.o matmul_cpu.o -o combined.o

../../build/bin/opt -O3 combined.o -o combined.o

../../build/bin/llc -O3 combined.o -o combined.s

$CCX -O3 combined.s -o a.out
