; RUN: mkdir -p %t/function-wise/

; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function foo && %t/function-wise/input-gen.function.foo.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.foo.run.a.out %t/function-wise/input-gen.function.foo.generate.a.out.input.0.bin) | FileCheck %s
; CHECK: OUTPUT [[o1:.*]]
; CHECK-NEXT: OUTPUT [[o2:.*]]
; CHECK: OUTPUT [[o1]]
; CHECK-NEXT: OUTPUT [[o2]]

; ModuleID = 'reduced.bc'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"OUTPUT %d\0A\00", align 1

define i32 @foo(ptr %parg) {
entry:
  %pstub = call ptr @bar()
  %arg = load i32, ptr %parg, align 8
  %stub = load i32, ptr %pstub, align 8
  %callstub = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %arg)
  %callarg = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %stub)
  ret i32 0
}

declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1
declare ptr @bar()

attributes #0 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 19.0.0git (git@github.com:llvm-ml/llvm-project.git 2fde12fb56ff350b5509b8f752c2d383594ee416)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
