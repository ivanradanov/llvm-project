; RUN: mkdir -p %t/function-wise/
;
; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function cmp && %t/function-wise/input-gen.function.cmp.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.cmp.run.a.out %t/function-wise/input-gen.function.cmp.generate.a.out.input.0.bin) | FileCheck %s
;
; TODO it would be nice if we could force the RT to give us different objects
; because that's when this coold break
; Check that we get the same result in input-gen and run
; CHECK: OUTPUT [[RES:.*]]
; CHECK: OUTPUT [[RES]]
;

; ModuleID = 'zero-size-obj-cmp.cpp'
source_filename = "zero-size-obj-cmp.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

@str = private unnamed_addr constant [17 x i8] c"OUTPUT NON-EQUAL\00", align 1
@str.2 = private unnamed_addr constant [13 x i8] c"OUTPUT EQUAL\00", align 1


; Function Attrs: nofree nounwind uwtable
define dso_local void @cmp(ptr noundef readnone %0, ptr noundef readnone %1) local_unnamed_addr #0 {
  %3 = icmp eq ptr %0, %1
  %4 = select i1 %3, ptr @str.2, ptr @str
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) %4)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"clang version 16.0.6 (Red Hat 16.0.6-2.module+el8.9.0+19521+190d7aba)"}
