; RUN: mkdir -p %t/function-wise/

; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function positive_offset_ptr && %t/function-wise/input-gen.function.positive_offset_ptr.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.positive_offset_ptr.run.a.out %t/function-wise/input-gen.function.positive_offset_ptr.generate.a.out.input.0.bin) | FileCheck %s --check-prefix=POSITIVE_PTR
; POSITIVE_PTR: OUTPUT [[out:.*]]
; POSITIVE_PTR: OUTPUT [[out]]

; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function negative_offset_ptr && %t/function-wise/input-gen.function.negative_offset_ptr.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.negative_offset_ptr.run.a.out %t/function-wise/input-gen.function.negative_offset_ptr.generate.a.out.input.0.bin) | FileCheck %s --check-prefix=NEGATIVE_PTR
; NEGATIVE_PTR: OUTPUT [[out:.*]]
; NEGATIVE_PTR: OUTPUT [[out]]

; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function positive_offset && %t/function-wise/input-gen.function.positive_offset.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.positive_offset.run.a.out %t/function-wise/input-gen.function.positive_offset.generate.a.out.input.0.bin) | FileCheck %s --check-prefix=POSITIVE
; POSITIVE: OUTPUT [[out:.*]]
; POSITIVE: OUTPUT [[out]]

; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function negative_offset && %t/function-wise/input-gen.function.negative_offset.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.negative_offset.run.a.out %t/function-wise/input-gen.function.negative_offset.generate.a.out.input.0.bin) | FileCheck %s --check-prefix=NEGATIVE
; NEGATIVE: OUTPUT [[out:.*]]
; NEGATIVE: OUTPUT [[out]]

; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function global_offset && %t/function-wise/input-gen.function.global_offset.generate.a.out %t/function-wise/ 0 1 && %t/function-wise/input-gen.function.global_offset.run.a.out %t/function-wise/input-gen.function.global_offset.generate.a.out.input.0.bin) | FileCheck %s --check-prefix=GLOBAL
; GLOBAL: OUTPUT [[out:.*]]
; GLOBAL: OUTPUT [[out]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"OUTPUT %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @positive_offset_ptr(ptr nocapture noundef readonly %ptr) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %ptr, i64 134
  %p = load ptr, ptr %add.ptr
  %ptr.offset = getelementptr inbounds i8, ptr %p, i64 52
  %0 = load i32, ptr %ptr.offset, align 4, !tbaa !5
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0)
  ret i32 %0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @negative_offset_ptr(ptr nocapture noundef readonly %ptr) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %ptr, i64 -48
  %p = load ptr, ptr %add.ptr
  %ptr.offset = getelementptr inbounds i8, ptr %p, i64 -276
  %0 = load i32, ptr %ptr.offset, align 4, !tbaa !5
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0)
  ret i32 %0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @positive_offset(ptr nocapture noundef readonly %ptr) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %ptr, i64 134
  %0 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0)
  ret i32 %0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @negative_offset(ptr nocapture noundef readonly %ptr) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %ptr, i64 -148
  %0 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0)
  ret i32 %0
}

@G = external global [163 x i32], align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @global_offset() local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr @G, i64 36
  %0 = load i32, ptr %add.ptr, align 4
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %0)
  ret i32 %0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

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
