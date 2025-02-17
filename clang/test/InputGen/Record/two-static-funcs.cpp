// XFAIL: *
// RUN: mkdir -p %t

// RUN: inputgen-minimize %s %S/Inputs/external_2.cpp %S/Inputs/external_1.cpp > %t/minimized.cpp
// RUN: %clangxx -c %t/minimized.cpp -o %t/minimized.o

#include "Inputs/static_func.h"

__attribute__((inputgen_entry))
int foo(int a, int b) {
    return external_1(a) + external_2(b);
}

int main() {
    foo(4, 5);
}
