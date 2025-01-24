; RUN: mkdir -p %t/function-wise/
;
; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function test && VERBOSE=1 %t/function-wise/input-gen.function.test.generate.a.out %t/function-wise/ 0 1) |& FileCheck %s --check-prefix=GGENN
; RUN: %t/function-wise/input-gen.function.test.run.a.out %t/function-wise/input-gen.function.test.generate.a.out.input.0.bin | FileCheck %s --check-prefix=RRUNN

; This test checks that we invalidate retry info correctly. In this example,
; when we make o2 point to 0 we need to invalidate the information about o4
; pointing to o3 as that happens after we changed o2 to nullptr and that
; decision may influence it
;
; __attribute__((noinline))
; void eqtest(const char *c, int *a, int *b) {
;   if (a == b)
;     printf("OUTPUT %s 0\n", c);
;   else
;     printf("OUTPUT %s 1\n", c);
; }
;
; void test(int *o1, int *o2, int *o3, int *o4) {
;   for (int i = 0; i < 100; i++)
;     eqtest("1", o3, o4);
;   for (int i = 0; i < 100; i++)
;     eqtest("2", o2, 0);
;   printf("OUTPUT o1 %p %d\n", o1, *o1);
;   printf("OUTPUT o2 %p\n", o2);
;   printf("OUTPUT o3 %p %d\n", o3, *o3);
;   printf("OUTPUT o4 %p %d\n", o4, *o4);
; }
;
; GGENN: Got 0 retry infos.
; GGENN: Got 1 retry infos.
; GGENN: Got 1 retry infos.
; GGENN: Got 2 retry infos.
; GEGNN-NOT: Got {{.*}} retry infos.
;
; RRUNN: OUTPUT o1 [[P1:.*]] [[V1:.*]]
; RRUNN: OUTPUT o2 (nil)
; RRUNN: OUTPUT o3 [[P2:.*]] [[V2:.*]]
; RRUNN: OUTPUT o4 [[P2]] [[V2]]




; ModuleID = 'rollback.cpp'
source_filename = "rollback.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"OUTPUT %s 0\0A\00", align 1
@.str.1 = private unnamed_addr constant [13 x i8] c"OUTPUT %s 1\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"1\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"2\00", align 1
@.str.4 = private unnamed_addr constant [17 x i8] c"OUTPUT o1 %p %d\0A\00", align 1
@.str.5 = private unnamed_addr constant [14 x i8] c"OUTPUT o2 %p\0A\00", align 1
@.str.6 = private unnamed_addr constant [17 x i8] c"OUTPUT o3 %p %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [17 x i8] c"OUTPUT o4 %p %d\0A\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @eqtest(ptr noundef %0, ptr noundef readnone %1, ptr noundef readnone %2) local_unnamed_addr #0 {
  %4 = icmp eq ptr %1, %2
  %5 = select i1 %4, ptr @.str, ptr @.str.1
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %5, ptr noundef %0)
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  br label %5

5:                                                ; preds = %4, %5
  %6 = phi i32 [ 0, %4 ], [ %7, %5 ]
  tail call void @eqtest(ptr noundef nonnull @.str.2, ptr noundef %2, ptr noundef %3)
  %7 = add nuw nsw i32 %6, 1
  %8 = icmp eq i32 %7, 100
  br i1 %8, label %17, label %5, !llvm.loop !3

9:                                                ; preds = %17
  %10 = load i32, ptr %0, align 4, !tbaa !6
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, ptr noundef nonnull %0, i32 noundef %10)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, ptr noundef %1)
  %13 = load i32, ptr %2, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef nonnull %2, i32 noundef %13)
  %15 = load i32, ptr %3, align 4, !tbaa !6
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef nonnull %3, i32 noundef %15)
  ret void

17:                                               ; preds = %5, %17
  %18 = phi i32 [ %19, %17 ], [ 0, %5 ]
  tail call void @eqtest(ptr noundef nonnull @.str.3, ptr noundef %1, ptr noundef null)
  %19 = add nuw nsw i32 %18, 1
  %20 = icmp eq i32 %19, 100
  br i1 %20, label %9, label %17, !llvm.loop !10
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !4, !5}
