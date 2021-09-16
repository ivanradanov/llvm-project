#!/bin/bash

set -x
set -e

CCX="../../build-clang/bin/clang++"
#OPT="../../build/bin/opt"
#OPT="../../build/bin/llvm-link"

$CCX -S main.cpp -O3 -emit-llvm

../../build/bin/opt main.ll -passes=cpucuda -o main.cpu.ll

#../../../debugir/build/debugir main.cpu.ll

../../build/bin/opt -O3 main.cpu.ll -o main.cpu.opt.ll
../../build/bin/llc -O3 main.cpu.opt.ll -o main.cpu.opt.s
$CCX -g -O3 main.cpu.opt.s -o a.out
