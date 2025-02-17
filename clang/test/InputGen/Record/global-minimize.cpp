// XFAIL: *

// RUN: inputgen-minimize %s | FileCheck %s

// RUN: mkdir -p %t
// RUN: %S/../../../scripts/inputgen_minimize.py --source-file %s --output-file %t/minimized.cpp
// RUN: %clangxx %t/minimized.cpp -o %t/minimized.a.out

// CHECK: #include <stdio.h>
// CHECK: int a
// CHECK: void foo
#include <stdio.h>

int a = 0;
__attribute__((inputgen_entry))
void foo() {
    printf("%d\n", a);
}
int main() {
    a = 5;
    foo();
    return 0;
}
