#!/bin/bash

set -x
set -e

CCX="../../build-clang/bin/clang++"
#OPT="../../build/bin/opt"
#OPT="../../build/bin/llvm-link"

$CCX -S main.cpp -O3 -emit-llvm

../../build/bin/opt main.ll -passes=cpucuda -o main.cpu.ll

../../../debugir/build/debugir main.cpu.ll

#$CCX -g -S test_main.cpp -O3 -emit-llvm

#../../build/bin/llvm-link main.ll matmul_cpu.dbg.ll -o combined.ll
#../../build/bin/llvm-link test_main.ll matmul_cpu.dbg.ll -o combined.bc
#../../build/bin/llvm-dis combined.bc -o combined.ll

#../../../debugir/build/debugir combined.ll


#../../build/bin/opt -O3 combined.o -o combined.o
#../../build/bin/llc -O3 combined.o -o combined.s
#$CCX -g -O3 combined.s -o a.out
