; RUN: mkdir -p %t
; RUN: mkdir -p %t/function-wise/
;
; RUN: (input-gen --instrumented-module-for-coverage --profiling-runtime-path=%libclang_rt_profile -g --verify --output-dir %t/ --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s && VERBOSE=1 %t/input-gen.module.generate.a.out %t/ 0 1 --name const_stub 0 && LLVM_PROFILE_FILE=%t/const_stub.prof VERBOSE=1 %t/input-gen.module.run.a.out %t/input-gen.module.generate.a.out.input.0.0.bin --name const_stub && llvm-profdata merge -o %t/const_stub.prof.merged %t/const_stub.prof && input-gen --instrumented-module-for-coverage --profiling-runtime-path=%libclang_rt_profile --profile-path %t/const_stub.prof.merged -g --verify --output-dir %t/ --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s && VERBOSE=1 %t/input-gen.module.generate.a.out %t/ 0 1 --name const_stub 0 && LLVM_PROFILE_FILE=%t/const_stub.prof2 VERBOSE=1 %t/input-gen.module.run.a.out %t/input-gen.module.generate.a.out.input.0.0.bin --name const_stub && llvm-profdata merge -o %t/const_stub.prof.merged %t/const_stub.prof %t/const_stub.prof2 && input-gen --instrumented-module-for-coverage --profiling-runtime-path=%libclang_rt_profile --profile-path %t/const_stub.prof.merged -g --verify --output-dir %t/ --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s && VERBOSE=1 %t/input-gen.module.generate.a.out %t/ 0 1 --name const_stub 0) | FileCheck %s --check-prefix=COVERAGE
;
; COVERAGE-DAG: GREATER
; COVERAGE-DAG: EQUAL
; COVERAGE-DAG: LESS


; RUN: input-gen -g --verify --output-dir %t --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s
; RUN: %S/run_all.sh %t
;
;
; RUN: input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function const_load
; RUN: VERBOSE=1 %t/function-wise/input-gen.function.const_load.generate.a.out %t/function-wise/ 0 1 2>&1 | FileCheck %s --check-prefix=CONST_LOAD
;
; CONST_LOAD: Access: {{.*}} Obj #0
; CONST_LOAD-DAG: BranchHint Kind 1 Signed 1 Frequency {{.*}} Val 1024
; CONST_LOAD-DAG: BranchHint Kind 2 Signed 1 Frequency {{.*}} Val 1024
; CONST_LOAD-DAG: BranchHint Kind 4 Signed 1 Frequency {{.*}} Val 1024
; CONST_LOAD-DAG: BranchHint Kind 5 Signed 1 Frequency {{.*}} Val 1024
;
;
; RUN: input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function const_arg
; RUN: VERBOSE=1 %t/function-wise/input-gen.function.const_arg.generate.a.out %t/function-wise/ 0 1 2>&1 | FileCheck %s --check-prefix=CONST_ARG
;
; CONST_ARG-DAG: BranchHint Kind 1 Signed 1 Frequency {{.*}} Val 256
; CONST_ARG-DAG: BranchHint Kind 2 Signed 1 Frequency {{.*}} Val 256
; CONST_ARG-DAG: BranchHint Kind 4 Signed 1 Frequency {{.*}} Val 256
; CONST_ARG-DAG: BranchHint Kind 5 Signed 1 Frequency {{.*}} Val 256
;
;
; RUN: input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function const_stub
; RUN: VERBOSE=1 %t/function-wise/input-gen.function.const_stub.generate.a.out %t/function-wise/ 0 1 2>&1 | FileCheck %s --check-prefix=CONST_STUB
;
; CONST_STUB-DAG: BranchHint Kind 1 Signed 1 Frequency {{.*}} Val 512
; CONST_STUB-DAG: BranchHint Kind 2 Signed 1 Frequency {{.*}} Val 512
; CONST_STUB-DAG: BranchHint Kind 4 Signed 1 Frequency {{.*}} Val 512
; COM: CONST_STUB-DAG: BranchHint Kind 5 Signed 1 Frequency {{.*}} Val 512
;
;

; ModuleID = 'branch-hints.cpp'
source_filename = "branch-hints.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

@str = private unnamed_addr constant [8 x i8] c"GREATER\00", align 1
@str.3 = private unnamed_addr constant [6 x i8] c"EQUAL\00", align 1
@str.4 = private unnamed_addr constant [5 x i8] c"LESS\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @greater() local_unnamed_addr #0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @equal() local_unnamed_addr #0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.3)
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @less() local_unnamed_addr #0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.4)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @const_arg(i32 noundef %0) local_unnamed_addr #1 {
  %2 = icmp sgt i32 %0, 256
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void @greater()
  br label %8

4:                                                ; preds = %1
  %5 = icmp eq i32 %0, 256
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  tail call void @equal()
  br label %8

7:                                                ; preds = %4
  tail call void @less()
  br label %8

8:                                                ; preds = %6, %7, %3
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @arg_arg(i32 noundef %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = icmp sgt i32 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void @greater()
  br label %9

5:                                                ; preds = %2
  %6 = icmp eq i32 %0, %1
  br i1 %6, label %7, label %8

7:                                                ; preds = %5
  tail call void @equal()
  br label %9

8:                                                ; preds = %5
  tail call void @less()
  br label %9

9:                                                ; preds = %7, %8, %4
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @const_load(ptr nocapture noundef readonly %0) local_unnamed_addr #1 {
  %2 = load i32, ptr %0, align 4, !tbaa !3
  %3 = icmp sgt i32 %2, 1024
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  tail call void @greater()
  br label %9

5:                                                ; preds = %1
  %6 = icmp eq i32 %2, 1024
  br i1 %6, label %7, label %8

7:                                                ; preds = %5
  tail call void @equal()
  br label %9

8:                                                ; preds = %5
  tail call void @less()
  br label %9

9:                                                ; preds = %7, %8, %4
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @load_arg(ptr nocapture noundef readonly %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = load i32, ptr %0, align 4, !tbaa !3
  %4 = icmp sgt i32 %3, %1
  br i1 %4, label %5, label %6

5:                                                ; preds = %2
  tail call void @greater()
  br label %10

6:                                                ; preds = %2
  %7 = icmp eq i32 %3, %1
  br i1 %7, label %8, label %9

8:                                                ; preds = %6
  tail call void @equal()
  br label %10

9:                                                ; preds = %6
  tail call void @less()
  br label %10

10:                                               ; preds = %8, %9, %5
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @const_stub() local_unnamed_addr #2 {
  %1 = tail call i32 (...) @stub1() #5
  %2 = icmp sgt i32 %1, 512
  br i1 %2, label %3, label %4

3:                                                ; preds = %0
  tail call void @greater()
  br label %8

4:                                                ; preds = %0
  %5 = icmp eq i32 %1, 512
  br i1 %5, label %6, label %7

6:                                                ; preds = %4
  tail call void @equal()
  br label %8

7:                                                ; preds = %4
  tail call void @less()
  br label %8

8:                                                ; preds = %6, %7, %3
  ret void
}

declare dso_local i32 @stub1(...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @stub_load(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i32, ptr %0, align 4, !tbaa !3
  %3 = tail call i32 (...) @stub1() #5
  %4 = icmp sgt i32 %3, %2
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  tail call void @greater()
  br label %10

6:                                                ; preds = %1
  %7 = icmp eq i32 %3, %2
  br i1 %7, label %8, label %9

8:                                                ; preds = %6
  tail call void @equal()
  br label %10

9:                                                ; preds = %6
  tail call void @less()
  br label %10

10:                                               ; preds = %8, %9, %5
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #4

attributes #0 = { nofree noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"clang version 16.0.6 (Red Hat 16.0.6-2.module+el8.9.0+19521+190d7aba)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
