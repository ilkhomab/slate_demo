program demo_zmatvec
  use mpi
  use iso_c_binding
  implicit none

  integer :: ierr, myrank, nprocs
  integer :: ictxt, nprow, npcol, myrow, mycol
  external :: blacs_get, blacs_gridinit, blacs_gridinfo, blacs_gridexit
  integer :: numroc
  external :: numroc

  integer :: n, mb, nb
  integer, parameter :: mbi = 256, nbi = 256

  integer :: lda_ll, ldb_ll, ldc_ll
  integer :: mlocA, nlocA, mlocB, nlocB, mlocC, nlocC

  integer(c_size_t) :: nelemA, nelemB, nelemC
  integer(c_size_t) :: bytesA, bytesB, bytesC

  type(c_ptr) :: A_dev, B_dev, C_dev
  complex(c_double_complex) :: sumA, sumB, sumC
  complex(c_double_complex) :: sum_globA,sum_globB,sum_globC
  complex(c_double_complex) :: A2(2), B2(2)


  interface
    function hip_alloc_bytes(nbytes) bind(C, name="hip_alloc_bytes") result(p)
      use iso_c_binding
      integer(c_size_t), value :: nbytes
      type(c_ptr) :: p
    end function

    subroutine hip_free(p) bind(C, name="hip_free")
      use iso_c_binding
      type(c_ptr), value :: p
    end subroutine

    subroutine fill_A_local_hip_dev(Adev, lda, mloc, nloc, n, mb, nb, p, q, myr, myc) &
      bind(C, name="fill_A_local_hip_dev")
      use iso_c_binding
      type(c_ptr), value :: Adev
      integer(c_int), value :: lda, mloc, nloc, n, mb, nb, p, q, myr, myc
    end subroutine

    subroutine fill_B_local_hip_dev(Bdev, ldv, mloc, nloc, n, mb, nb, p, q, myr, myc) &
      bind(C, name="fill_B_local_hip_dev")
      use iso_c_binding
      type(c_ptr), value :: Bdev
      integer(c_int), value :: ldv, mloc, nloc, n, mb, nb, p, q, myr, myc
    end subroutine

    subroutine print_gpu_info(rank) bind(C, name="print_gpu_info")
      use iso_c_binding
      integer(c_int), value :: rank
    end subroutine

    subroutine gpu_sum_complex(devptr, n, sum_out) bind(C, name="gpu_sum_complex")
      use iso_c_binding
      type(c_ptr), value :: devptr
      integer(c_size_t), value :: n
      complex(c_double_complex), intent(out) :: sum_out
    end subroutine

    subroutine gpu_get_first2_complex(devptr, n, out2) bind(C, name="gpu_get_first2_complex")
      use iso_c_binding
      type(c_ptr), value :: devptr
      integer(c_size_t), value :: n
      complex(c_double_complex), intent(out) :: out2(2)
    end subroutine

    ! NEW: SLATE wrapper (defined in slate_wrap.cpp)
    subroutine slate_zgemm_from_devices(n, mb, nb, p, q, dA, lda, dB, ldb, dC, ldc, fcomm) &
      bind(C, name="slate_zgemm_from_devices")
      use iso_c_binding
      integer(c_long_long), value :: n, mb, nb
      integer(c_int), value :: p, q
      type(c_ptr), value :: dA, dB, dC
      integer(c_long_long), value :: lda, ldb, ldc
      integer(c_int), value :: fcomm
    end subroutine
  end interface

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  n  = 20*1024
  mb = mbi
  nb = nbi

  call choose_grid(nprocs, nprow, npcol)
  call blacs_get(-1, 0, ictxt)
  call blacs_gridinit(ictxt, 'Col', nprow, npcol)
  call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)

  mlocA = numroc(n, mbi, myrow, 0, nprow)
  nlocA = numroc(n, nbi, mycol, 0, npcol)
  mlocB = numroc(n, mbi, myrow, 0, nprow)
  nlocB = numroc(1, nbi, mycol, 0, npcol)

  ! C has same distribution as B (n x 1)
  mlocC = numroc(n, mbi, myrow, 0, nprow)
  nlocC = numroc(1, nbi, mycol, 0, npcol)

  lda_ll = mlocA
  ldb_ll = mlocB
  ldc_ll = mlocC

  nelemA = int(max(1, lda_ll), c_size_t) * int(max(0, nlocA), c_size_t)
  nelemB = int(max(1, ldb_ll), c_size_t) * int(max(0, nlocB), c_size_t)
  nelemC = int(max(1, ldc_ll), c_size_t) * int(max(0, nlocC), c_size_t)

  ! allocate >= 1 element to keep hipMalloc happy and avoid issues on ranks with nloc==0
  bytesA = max(nelemA, 1_c_size_t) * c_sizeof((0.0d0, 0.0d0))
  bytesB = max(nelemB, 1_c_size_t) * c_sizeof((0.0d0, 0.0d0))
  bytesC = max(nelemC, 1_c_size_t) * c_sizeof((0.0d0, 0.0d0))

!  print *, 'rank', myrank, 'n=', n, 'mb=', mb, 'nb=', nb
!  print *, 'rank', myrank, 'nprow=', nprow, 'npcol=', npcol, 'myrow=', myrow, 'mycol=', mycol
!  print *, 'rank', myrank, 'mlocA=', mlocA, 'nlocA=', nlocA, 'lda=', lda_ll, 'nelemA=', nelemA
!  print *, 'rank', myrank, 'mlocB=', mlocB, 'nlocB=', nlocB, 'ldb=', ldb_ll, 'nelemB=', nelemB
!  print *, 'rank', myrank, 'mlocC=', mlocC, 'nlocC=', nlocC, 'ldc=', ldc_ll, 'nelemC=', nelemC

  call print_gpu_info(myrank)

  A_dev = hip_alloc_bytes(bytesA)
  B_dev = hip_alloc_bytes(bytesB)
  C_dev = hip_alloc_bytes(bytesC)

  if (mlocA > 0 .and. nlocA > 0) then
    call fill_A_local_hip_dev(A_dev, lda_ll, mlocA, nlocA, n, mbi, nbi, nprow, npcol, myrow, mycol)
  end if
  if (mlocB > 0 .and. nlocB > 0) then
    call fill_B_local_hip_dev(B_dev, ldb_ll, mlocB, nlocB, n, mbi, nbi, nprow, npcol, myrow, mycol)
  end if

  A2 = (0.0d0, 0.0d0)
  B2 = (0.0d0, 0.0d0)
  
  call gpu_get_first2_complex(A_dev, nelemA, A2)
  call gpu_get_first2_complex(B_dev, nelemB, B2)
  
  if (nelemA >= 1_c_size_t) print *, 'rank', myrank, 'A(1)=', A2(1)
  if (nelemA >= 2_c_size_t) print *, 'rank', myrank, 'A(2)=', A2(2)
  
  if (nelemB >= 1_c_size_t) print *, 'rank', myrank, 'B(1)=', B2(1)
  if (nelemB >= 2_c_size_t) print *, 'rank', myrank, 'B(2)=', B2(2)


  if (nelemA > 0_c_size_t) then
    call gpu_sum_complex(A_dev, nelemA, sumA)
  else
    sumA = (0.0d0, 0.0d0)
  end if

  if (nelemB > 0_c_size_t) then
    call gpu_sum_complex(B_dev, nelemB, sumB)
  else
    sumB = (0.0d0, 0.0d0)
  end if

  print *, 'rank', myrank, 'sum_locA =', sumA
  print *, 'rank', myrank, 'sum_locB =', sumB

  ! ---- SLATE: C := A * B (B is n x 1) ----
  call slate_zgemm_from_devices( int(n, c_long_long), int(mb, c_long_long), int(nb, c_long_long), &
                                 nprow, npcol, A_dev, int(lda_ll, c_long_long), &
                                 B_dev, int(ldb_ll, c_long_long), &
                                 C_dev, int(ldc_ll, c_long_long), &
                                 MPI_COMM_WORLD )

  if (nelemC > 0_c_size_t) then
    call gpu_sum_complex(C_dev, nelemC, sumC)
  else
    sumC = (0.0d0, 0.0d0)
  end if
  print *, 'rank', myrank, 'sum_locC =', sumC

  call MPI_Allreduce(sumA, sum_globA, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(sumB, sum_globB, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(sumC, sum_globC, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD, ierr)

  if (myrank == 0) then
    print *, 'GLOBAL sum(A) =', sum_globA
    print *, 'GLOBAL sum(B) =', sum_globB
    print *, 'GLOBAL sum(C) =', sum_globC
  end if

  call hip_free(A_dev)
  call hip_free(B_dev)
  call hip_free(C_dev)

  call blacs_gridexit(ictxt)
  call MPI_Finalize(ierr)

contains

  subroutine choose_grid(nprocs, nprow, npcol)
    implicit none
    integer, intent(in)  :: nprocs
    integer, intent(out) :: nprow, npcol
    integer :: p
    p = int(sqrt(real(nprocs)))
    do while (p > 1 .and. mod(nprocs, p) /= 0)
      p = p - 1
    end do
    nprow = p
    npcol = nprocs / p
  end subroutine choose_grid

end program demo_zmatvec

