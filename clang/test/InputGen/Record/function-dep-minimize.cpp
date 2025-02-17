// RUN: inputgen-minimize %s %S/Inputs/external_func.cpp

// RUN: mkdir -p %t
// RUN: %S/../../../scripts/inputgen_minimize.py %s %S/Inputs/external_func.cpp --output-file %t/minimized.cpp
// RUN: %clangxx %t/minimized.cpp -o %t/minimized.a.out

#include "Inputs/external_func.h"

#include "Inputs/function-dep.h"

using int_h = int;

using int_x = int;
using int_a = int_x;
typedef int int_y;
typedef int int_unused;

namespace one {
    struct UnusedStruct {
        int usfoo(int a, int_h b) {
            return a + b;
        }
    };
    struct Struct3 {
        int s3foo(int a, int_h b) {
            return foo_in_header(a, b);
        }
    };
}
typedef int_y int_b;
using int_unused2 = int;
namespace one {
    class UnusedClass {
        int ucfoo(int a, int_h b) {
            return a + b;
        }
    };
    class Class2 {
        private:
            int c1priv;
        public:
            int c1pub;
        int c2foo(int a, int_h b) {
            return external_foo(a, b);
        }
    };
    class Class1 {
        public:
            int c1bar(int a, int b);
            int c1baz(int a, int b);
        static int c1foo(int a, int_h b) {
            return Class2().c2foo(a, b);
        }
    };
    int Class1::c1bar(int a, int b) {
        return a + b;
    }
    int Class1::c1baz(int a, int b) {
        return a + b;
    }

}
namespace named {
typedef int int_c;
using int_d = int;
using int_f = int;
using int_unused3 = int_unused2;
using int_unused4 = int;
int_c qux(int_a a, int_b b) {
    using int_e = int_d;
    using int_g = int_f;
    using int_h = int_d;
    int_e r = one::Class1::c1foo(a, b) + one::Struct3().s3foo(a, b) + one::Class1().c1bar(a, b);
    return r;
}
int_unused2 qux_unused(int_a a, int_b b) {
    int_d r = a + b;
    return r;
}
} // namespace named

namespace {
extern "C++" {
    int unused_baz(int a, int b) { return 0; }
    int baz(int a, int b) {
        return named::qux(a, b);
    }
}
} // anonymous namespace

int bar(int a, int b);
__attribute__((inputgen_entry))
int foo(int a, int b) {
    return bar(a, b) + named::qux(a, b);
}

int bar(int a, int b) {
    return baz(a, b);
}

int main() {
    foo(4, 5);
}
