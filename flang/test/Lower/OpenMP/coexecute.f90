! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtarget_teams_coexecute
subroutine target_teams_coexecute()
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.coexecute
  !$omp target teams coexecute
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end target teams coexecute
end subroutine target_teams_coexecute

! CHECK-LABEL: func @_QPteams_coexecute
subroutine teams_coexecute()
  ! CHECK: omp.teams
  ! CHECK: omp.coexecute
  !$omp teams coexecute
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end teams coexecute
end subroutine teams_coexecute
