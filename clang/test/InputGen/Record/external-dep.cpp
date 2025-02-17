// RUN: mkdir -p %t

// RUN: inputgen-minimize %s %S/Inputs/external_func.cpp > %t/minimized.cpp
// RUN: %clangxx -c %t/minimized.cpp -o %t/minimized.o

#include "Inputs/external_func.h"

__attribute__((inputgen_entry))
int foo(int a, int b) {
    return external_foo(a, b);
}

int main() {
    foo(4, 5);
}
