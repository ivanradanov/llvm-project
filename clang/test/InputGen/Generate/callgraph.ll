; RUN: mkdir -p %t
; RUN: input-gen -g --verify --output-dir %t --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s
; RUN: %S/run_all.sh %t


define dso_local void @func1() {
; CHECK-LABEL: @func1(
;
  %1 = icmp ne i32 0, 0
  br i1 %1, label %2, label %3

2:                                                ; preds = %0
  call void @func2(i1 false)
  br label %3

3:                                                ; preds = %2, %0
  call void () @func3()
  ret void
}

declare void @func3()
define internal void @func4() {
; CHECK-LABEL: @func4(
;
  call void @func3()
  ret void
}
define internal void @internal_good() {
; CHECK-LABEL: @internal_good(
;
  call void @void(ptr @func4)
  ret void
}

define dso_local void @func2(i1 %c) {
; UPTO2-LABEL: @func2(
;
; LIMI0-LABEL: @func2(
;
  %f = select i1 %c, ptr @internal_good, ptr @func4
  call void %f()
  ret void
}


define void @func5(i32 %0) {
; UPTO2-LABEL: @func5(
;
; LIMI0-LABEL: @func5(
;
  %2 = icmp ne i32 %0, 0
  %3 = select i1 %2, ptr @func4, ptr @func3
  call void () %3()
  ret void
}

define i32 @musttailCall(i32 %0) {
; CHECK-LABEL: @musttailCall(
;
  %2 = icmp ne i32 %0, 0
  %3 = select i1 %2, ptr @func4, ptr @func3
  %c = musttail call i32 (i32) %3(i32 0)
  ret i32 %c
}

declare i32 @retI32()
declare void @takeI32(i32)
declare float @retFloatTakeFloat(float)
; This callee is always filtered out because of the noundef argument
declare float @retFloatTakeFloatFloatNoundef(float, float noundef)
declare void @void()

define i32 @non_matching_fp1(i1 %c1, i1 %c2, i1 %c) {
; UNLIM-LABEL: @non_matching_fp1(
;
; LIMI2-LABEL: @non_matching_fp1(
;
; LIMI0-LABEL: @non_matching_fp1(
;
  %fp1 = select i1 %c1, ptr @retI32, ptr @takeI32
  %fp2 = select i1 %c2, ptr @retFloatTakeFloat, ptr @void
  %fp = select i1 %c, ptr %fp1, ptr %fp2
  %call = call i32 %fp(i32 42)
  ret i32 %call
}

define i32 @non_matching_fp1_noundef(i1 %c1, i1 %c2, i1 %c) {
; UNLIM-LABEL: @non_matching_fp1_noundef(
;
; LIMI2-LABEL: @non_matching_fp1_noundef(
;
; LIMI0-LABEL: @non_matching_fp1_noundef(
;
  %fp1 = select i1 %c1, ptr @retI32, ptr @takeI32
  %fp2 = select i1 %c2, ptr @retFloatTakeFloatFloatNoundef, ptr @void
  %fp = select i1 %c, ptr %fp1, ptr %fp2
  %call = call i32 %fp(i32 42)
  ret i32 %call
}

define void @non_matching_fp2(i1 %c1, i1 %c2, i1 %c, ptr %unknown) {
; OUNLM-LABEL: @non_matching_fp2(
;
; LIMI2-LABEL: @non_matching_fp2(
;
; LIMI0-LABEL: @non_matching_fp2(
;
; CWRLD-LABEL: @non_matching_fp2(
;
  %fp1 = select i1 %c1, ptr @retI32, ptr @takeI32
  %fp2 = select i1 %c2, ptr @retFloatTakeFloat, ptr %unknown
  %fp = select i1 %c, ptr %fp1, ptr %fp2
  call void %fp()
  ret void
}

define i32 @non_matching_unknown(i1 %c, ptr %fn) {
; OUNLM-LABEL: @non_matching_unknown(
;
; LIMI2-LABEL: @non_matching_unknown(
;
; LIMI0-LABEL: @non_matching_unknown(
;
; CWRLD-LABEL: @non_matching_unknown(
;
  %fp = select i1 %c, ptr @retI32, ptr %fn
  %call = call i32 %fp(i32 42)
  ret i32 %call
}

; This function is used in a "direct" call but with a different signature.
; We check that it does not show up above in any of the if-cascades because
; the address is not actually taken.
declare void @usedOnlyInCastedDirectCall(i32)
define void @usedOnlyInCastedDirectCallCaller() {
; CHECK-LABEL: @usedOnlyInCastedDirectCallCaller(
;
  call void @usedOnlyInCastedDirectCall()
  ret void
}

define internal void @usedByGlobal() {
; CHECK-LABEL: @usedByGlobal(
;
  ret void
}
@G = global ptr @usedByGlobal

define void @broker(ptr %unknown) !callback !0 {
  call void %unknown()
  ret void
}

define void @func6() {
  call void @broker(ptr @func3)
  ret void
}

; Cannot be internal_good as it is internal and we see all uses.
; Can be func4 since it escapes.
define void @func7(ptr %unknown) {
; UPTO2-LABEL: @func7(
;
; LIMI0-LABEL: @func7(
;
  call void %unknown(), !callees !2
  ret void
}

define void @func8(ptr %another_unknown) {
  call void @broker(ptr %another_unknown)
  ret void
}

define void @func9(ptr %mem) {
  %1 = load ptr, ptr %mem
  call void @broker(ptr %1)
  ret void
}

;define void @as_cast(ptr %arg) {
;; OWRDL-LABEL: @as_cast(
;;
;; CWRLD-LABEL: @as_cast(
;;
;  %fp = load ptr, ptr %arg, align 8
;  tail call void %fp()
;  ret void
;}

!0 = !{!1}
!1 = !{i64 0, i1 false}
!2 = !{ptr @func3, ptr @func4}
