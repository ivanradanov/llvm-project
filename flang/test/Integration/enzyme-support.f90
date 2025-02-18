! RUN: %flang_fc1 -O2 -emit-llvm %s -o - | FileCheck %s

! CHECK:   %[[FPTR:.*]] = call ptr @_QFPf__enzyme_truncate_add_func.1(ptr {{.*}} @_QFPadd_func, i32 32, i32 8, i32 5)
! CHECK:   %{{.*}} = call float %[[FPTR]]({{.*}}, {{.*}})

program pointer_to_function
  implicit none

  real :: num1, num2, result
  interface
     function add(a, b)
       real, intent(in) :: a, b
       real :: add
     end function add
  end interface

  ! Declare a function pointer
  procedure(add), pointer :: func_ptr

  ! Input two real numbers
  print *, 'Enter the first number:'
  read *, num1

  print *, 'Enter the second number:'
  read *, num2

  result = f__enzyme_truncate_add_func(32, 8, 5, num1, num2)

  ! Output the result
  print *, 'The sum of ', num1, ' and ', num2, ' is: ', result

contains

  ! Function to add two real numbers
  function add_func(a, b) result(c)
    real, intent(in) :: a, b
    real :: c
    c = a + b
  end function add_func

  function f__enzyme_truncate_add_func(from, to_m, to_e, a, b) result(c)
    integer :: from, to_m, to_e
    real, intent(in) :: a, b
    real :: c
    c = add_func(a, b)
  end function f__enzyme_truncate_add_func

end program pointer_to_function
