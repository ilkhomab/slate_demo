program demo_zmatvec_serial
  ! Single-process CPU demo: repeated dense complex mat-vec
  ! C <- A * B, using explicit loops (no BLAS/GEMM), with A,B generated
  ! via the same deterministic valA / valB formulas as in the MPI code.

  use iso_c_binding, only: c_double_complex
  implicit none

  integer, parameter :: n = 20*1024         ! global size
  integer           :: i, j, iter
  complex(c_double_complex), allocatable :: B(:), C(:)
  complex(c_double_complex) :: s
  integer :: count_rate, count_start, count_end
  real(8) :: elapsed
  complex(c_double_complex) :: sumA, sumB, sumC

  ! Allocate vectors
  allocate(B(n), C(n))

  ! Fill initial B from global formula
  do i = 1, n
    B(i) = valB(i, n)
  end do

  ! Time the repeated mat-vecs
  call system_clock(count_rate = count_rate)
  call system_clock(count_start)


     ! C = A * B (manual mat-vec, no BLAS)
     sumA = (0.0d0,0.d0)
     do i = 1, n
        s = (0.0d0, 0.0d0)
        do j = 1, n
           s = s + valA(i, j, n) * B(j)
           sumA=sumA+valA(i,j,n)
        end do
        C(i) = s
     end do


  call system_clock(count_end)
  elapsed = real(count_end - count_start, 8) / real(count_rate, 8)

  ! Simple sanity output
  sumB = (0.0d0,0.d0)
  sumC = (0.0d0,0.d0)
  do i = 1, n
    sumB = sumB + B(i)   ! real part only for a scalar summary
    sumC = sumC + C(i)
  end do

  print *, 'First 2 entries of B matrix:'
  print *, B(1:2)

  print *, 'First 3 entries of final C:'
  print *, C(1:3)
  print *, 'Sum of A:', sumA
  print *, 'Sum of B:', sumB
  print *, 'Sum of C:', sumC
  print *, 'Elapsed time (s) for 6 mat-vecs = ', elapsed

  deallocate(B, C)

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

  pure function valB(gi, n) result(z)
    integer, intent(in) :: gi, n
    complex(c_double_complex) :: z
    integer :: r1, r2
    r1 = mod(13*gi, 1024)
    r2 = mod(17*gi, 1024)
    z  = cmplx( dble(r1)/1024.d0, -dble(r2)/1024.d0, kind=8 )
  end function valB

end program demo_zmatvec_serial

