program demo_zmatmat_serial
  ! Single-process CPU demo: repeated dense complex matrix-matrix multiply
  ! C <- A * B, using explicit loops (no BLAS/GEMM), with A,B generated
  ! via deterministic valA / valB formulas.

  use iso_c_binding, only: c_double_complex
  implicit none

  ! NOTE: n = 20*1024 => ~6.7 GB per matrix of complex*16; 3 matrices ~20 GB.
  ! Reduce n for practical runs on typical nodes.
  integer, parameter :: n = 2048  ! was 20*1024 for the vector case
  integer           :: i, j, k, iter
  complex(c_double_complex), allocatable :: A(:,:), B(:,:), C(:,:)
  complex(c_double_complex), allocatable :: X(:), Y(:)
  complex(c_double_complex) :: s
  integer :: count_rate, count_start, count_end
  real(8) :: elapsed
  real(8) :: sumB, sumC

  ! Allocate matrices
  allocate(A(n,n), B(n,n), C(n,n))

  allocate(X(n),Y(n))

  X(1:n)=(1.d0,1.d0)

  ! Fill A and B from global formulas
  do j = 1, n
    do i = 1, n
      A(i,j) = valA(i, j, n)
      B(i,j) = valB(i, j, n)
    end do
  end do

  ! Time the repeated mat-mats
  call system_clock(count_rate = count_rate)
  call system_clock(count_start)

  do iter = 0, 2!5
     print *, 'Iteration ', iter

     ! C = A * B (manual matrix-matrix multiply, no BLAS)
     do j = 1, n
        do i = 1, n
           s = (0.0d0, 0.0d0)
           do k = 1, n
              s = s + A(i, k) * B(k, j)
           end do
           C(i, j) = s
        end do
     end do

     Y=matmul(C,X)


     ! For the next iteration, B <- C
     B = C
     X=Y
  end do

  call system_clock(count_end)
  elapsed = real(count_end - count_start, 8) / real(count_rate, 8)

  ! Simple sanity output
  sumB = 0.0d0
  sumC = 0.0d0
  do j = 1, n
    do i = 1, n
      sumB = sumB + dble(B(i,j))   ! real part only for a scalar summary
      sumC = sumC + dble(C(i,j))
    end do
  end do

  print *, 'Top-left 3x3 block of final C (if n>=3):'
  if (n >= 3) then
    do i = 1, 3
      print *, C(i,1:3)
    end do
  end if
  print *, 'Sum of real parts of B:', sumB
  print *, 'Sum of real parts of C:', sumC
  print *, 'Elapsed time (s) for 6 mat-mats = ', elapsed

  print*,'PrintY', Y(1:3)

  deallocate(A, B, C)

contains

  ! ---------- Cheap deterministic test data (integer arithmetic only) ----------
  pure function valA(gi, gj, n) result(z)
    integer, intent(in) :: gi, gj, n
    complex(c_double_complex) :: z
    integer :: r1, r2
    r1 = mod(3*gi + 5*gj, 1024)
    r2 = mod(7*gi - 11*gj, 1024)
    z  = cmplx( dble(r1)/1024.d0, dble(r2)/1024.d0, kind=8 )
  end function valA

  pure function valB(gi, gj, n) result(z)
    integer, intent(in) :: gi, gj, n
    complex(c_double_complex) :: z
    integer :: r1, r2
    r1 = mod(13*gi + 19*gj, 1024)
    r2 = mod(17*gi - 23*gj, 1024)
    z  = cmplx( dble(r1)/1024.d0, -dble(r2)/1024.d0, kind=8 )
  end function valB

end program demo_zmatmat_serial

