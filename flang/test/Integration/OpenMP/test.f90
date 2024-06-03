
program test
    implicit none

    integer, parameter :: N = 20480
    !integer, parameter :: N = 16
    real :: a = 7, b
    real, dimension(:, :), allocatable :: x
    real, dimension(:, :), allocatable :: y
    real, dimension(:, :), allocatable :: z

    allocate(x(N, N))
    allocate(y(N, N))
    allocate(z(N, N))

    ! x = (3, 1)
    ! y = (2, -1)
    x = 3
    y = 2
    z = 0

    write (*, '(A)') 'calling axpy'
    b = abs(coexecute_a(x, y, z, N, a))

    deallocate(x)
    deallocate(y)
    deallocate(z)

contains
function coexecute_a(x, y, z, n, a) result(sum_less)
  use omp_lib
  implicit none
  integer :: n, i, j
  real :: sum_less, a
  real, dimension(n, n) :: x, y, z
  double precision :: ostart, oend

  write (*,*) 'n before', n
  write (*,*) 'a before', a
  write (*,*) 'z(1,1) before', z(1,1)
  write (*,*) 'checksum before', sum(z(1:n, 1:n))

  ostart = omp_get_wtime()

  !$omp target teams coexecute
    y = a * x + y
    a = 2
    z = sqrt(y) + x * a
  !$omp end target teams coexecute

  oend = omp_get_wtime()


  write (*,*) 'n after', n
  write (*,*) 'a after', a
  write (*,*) 'z(1,1) after', z(1,1)
  write (*,*) 'checksum after', sum(z(1:n, 1:n))

  ! do i = 1, n
  !    do j = 1, n
  !       write (*,*) 'z', i, " ", j, " ", z(i,j)
  !    end do
  ! end do

  print *, 'Time: ', oend-ostart, 'seconds.'

  sum_less = sum(z(1:n/2,1:n/3) - 2) / ( n * n)

end function coexecute_a
end program test
