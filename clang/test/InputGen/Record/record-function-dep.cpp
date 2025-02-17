// RUN: mkdir -p %t
// RUN: %clangxx -g -O2 %s -mllvm --input-gen-mode=record -o %t/record.a.out -linputgen.record
// RUN: %clangxx -g -O2 %s -mllvm --input-gen-mode=replay -rdynamic -o %t/replay.a.out -linputgen.replay
// RUN: rm %t/input.bin || true
// RUN: INPUT_RECORD_FILENAME=%t/input.bin %t/record.a.out | FileCheck %s
// RUN: %t/replay.a.out --input %t/input.bin | FileCheck %s

// RUN: %S/../../../scripts/inputgen_minimize.py %s --embed-input-file %t/input.bin --output-file %t/minimized.cpp
// RUN: %clangxx %t/minimized.cpp -o %t/minimized.a.out
// RUN: %t/minimized.a.out | FileCheck %s

// RUN: %S/../../../scripts/inputgen_minimize.py %s --output-file %t/minimized.cpp
// RUN: %clangxx %t/minimized.cpp -o %t/minimized.a.out
// RUN: %t/minimized.a.out --input %t/input.bin | FileCheck %s

// CHECK: Sum: 495

#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>

#define N 10

int bar(int a, int b) {
    return a + b;
}

__attribute__((inputgen_entry))
int foo(int *a, int *b, int *c, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    c[i] = bar(a[i], b[i] * n);
    sum += c[i];
  }
  printf("Sum: %d\n", sum);
  return sum;
}

int main() {
  int *a = (int *)malloc(N * sizeof(*a));
  int *b = (int *)malloc(N * sizeof(*b));
  int *c = (int *)malloc(N * sizeof(*c));

  for (int i = 0; i < N; i++) {
    a[i] = b[i] = i % 10;
  }

  int d = foo(a, b, c, N);
  printf("Output: %d\n", d);

  free(a);
  free(b);
  free(c);
  return 0;
}
