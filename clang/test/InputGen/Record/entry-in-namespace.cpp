// RUN: inputgen-minimize %s | FileCheck %s

// CHECK:     bar::foo();

int puts(const char *);

namespace bar {
__attribute__((inputgen_entry))
void foo() {
  puts("wow");
}
}
