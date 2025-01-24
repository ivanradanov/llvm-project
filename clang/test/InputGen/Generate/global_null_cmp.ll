; RUN: mkdir -p %t/function-wise/
;
; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function test && %t/function-wise/input-gen.function.test.generate.a.out %t/function-wise/ 0 1) | FileCheck %s --check-prefix=GGENN
; RUN: %t/function-wise/input-gen.function.test.run.a.out %t/function-wise/input-gen.function.test.generate.a.out.input.0.bin | FileCheck %s --check-prefix=RRUNN
;
; GGENN-NOT: OUTPUT 0
; RRUNN-NOT: OUTPUT 0
;
; Tests that we do not accidentally relocate a global to be null
;
; __attribute__((noinline))
; void eqtest(int *a) {
;   if (a == 0)
;     puts("OUTPUT 0");
;   else
;     puts("OUTPUT 1");
;
; }
;
; void test() {
;   for (int i = 0; i < 1000; i++) {
;     eqtest(&a);
;   }
; }
;
; ModuleID = 'global_null_cmp.cpp'
source_filename = "global_null_cmp.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"OUTPUT 0\00", align 1
@.str.1 = private unnamed_addr constant [9 x i8] c"OUTPUT 1\00", align 1
@a = dso_local global i32 0, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @eqtest(ptr noundef readnone %0) local_unnamed_addr #0 {
  %2 = icmp eq ptr %0, null
  %3 = select i1 %2, ptr @.str, ptr @.str.1
  %4 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %3)
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test() local_unnamed_addr #2 {
  br label %2

1:                                                ; preds = %2
  ret void

2:                                                ; preds = %0, %2
  %3 = phi i32 [ 0, %0 ], [ %4, %2 ]
  tail call void @eqtest(ptr noundef nonnull @a)
  %4 = add nuw nsw i32 %3, 1
  %5 = icmp eq i32 %4, 100
  br i1 %5, label %1, label %2, !llvm.loop !3
}

attributes #0 = { nofree noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"clang version 16.0.6 (Red Hat 16.0.6-2.module+el8.9.0+19521+190d7aba)"}
!3 = distinct !{!3, !4, !5}
!4 = !{!"llvm.loop.mustprogress"}
!5 = !{!"llvm.loop.unroll.disable"}
