program demo_zmatvec
  use mpi
  use iso_c_binding
  implicit none

  integer, parameter :: root = 0
  integer :: ierr, myrank, nprocs
  integer :: ictxt, nprow, npcol, myrow, mycol
  external :: blacs_get, blacs_gridinit, blacs_gridinfo, blacs_gridexit
  integer :: numroc
  external :: numroc

  ! global size & blocking
  integer :: n, mb, nb
  integer, parameter :: mbi = 256, nbi = 256

  ! local sizes & leading dims
  integer :: lda_ll, ldb_ll, ldc_ll
  integer :: mlocA, nlocA, mlocB, nlocB, mlocC, nlocC
  integer(c_int64_t) :: lenA, lenB, lenC

  ! Device pointers (GPU-resident)
  type(c_ptr) :: dA, dB, dC

  ! For printing sums (computed on GPU, copied back as 1 scalar)
  complex(c_double_complex) :: sumA_h, sumB_h, sumC_h

  ! ----------------- interfaces -----------------
  interface
    subroutine choose_grid(nprocs, nprow, npcol) bind(C, name="choose_grid_f")
      import :: c_int
      integer(c_int), value :: nprocs
      integer(c_int) :: nprow, npcol
    end subroutine

    subroutine hip_init_and_print(myrank) bind(C, name="hip_init_and_print")
      import :: c_int
      integer(c_int), value :: myrank
    end subroutine

    subroutine gpu_alloc_complex(n, dptr) bind(C, name="gpu_alloc_complex")
      import :: c_int64_t, c_ptr
      integer(c_int64_t), value :: n
      type(c_ptr) :: dptr
    end subroutine

    subroutine gpu_free(dptr) bind(C, name="gpu_free")
      import :: c_ptr
      type(c_ptr), value :: dptr
    end subroutine

    subroutine fill_A_local_gpu(dA, lda_i, n_i, mb_i, nb_i, p, q, myr, myc) &
      bind(C, name="fill_A_local_gpu")
      import :: c_ptr, c_int, c_int64_t
      type(c_ptr), value :: dA
      integer(c_int), value :: lda_i, n_i, mb_i, nb_i, p, q, myr, myc
    end subroutine

    subroutine fill_B_local_gpu(dB, ldb_i, n_i, mb_i, nb_i, p, q, myr, myc) &
      bind(C, name="fill_B_local_gpu")
      import :: c_ptr, c_int
      type(c_ptr), value :: dB
      integer(c_int), value :: ldb_i, n_i, mb_i, nb_i, p, q, myr, myc
    end subroutine

    subroutine device_sum_z(n, dX, out_sum_host) bind(C, name="device_sum_z")
      import :: c_int64_t, c_ptr, c_double_complex
      integer(c_int64_t), value :: n
      type(c_ptr), value :: dX
      complex(c_double_complex) :: out_sum_host
    end subroutine

    subroutine slate_zgemm_from_devices(n, mb, nb, p, q, dA, lda, dB, ldb, dC, ldc, fcomm) &
      bind(C, name="slate_zgemm_from_devices")
      import :: c_int64_t, c_int, c_ptr
      integer(c_int64_t), value :: n
      integer(c_int64_t), value :: mb, nb
      integer(c_int), value :: p, q
      type(c_ptr), value :: dA, dB, dC
      integer(c_int64_t), value :: lda, ldb, ldc
      integer(c_int), value :: fcomm
    end subroutine
  end interface
  ! ------------------------------------------------

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! pick GPU and print mapping (each rank prints)
  call hip_init_and_print(myrank)

  n  = 20*1024
  mb = mbi
  nb = nbi

  call choose_grid(nprocs, nprow, npcol)
  call blacs_get(-1, 0, ictxt)
  call blacs_gridinit(ictxt, 'Row', nprow, npcol)
  call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)

  mlocA = numroc(n, mbi, myrow, 0, nprow)
  nlocA = numroc(n, nbi, mycol, 0, npcol)
  mlocB = numroc(n, mbi, myrow, 0, nprow)
  nlocB = numroc(1, nbi, mycol, 0, npcol)
  mlocC = numroc(n, mbi, myrow, 0, nprow)
  nlocC = numroc(1, nbi, mycol, 0, npcol)

  lda_ll = mlocA
  ldb_ll = mlocB
  ldc_ll = mlocC

  lenA = int(max(1, lda_ll), c_int64_t) * int(max(0, nlocA), c_int64_t)
  lenB = int(max(1, ldb_ll), c_int64_t) * int(max(0, nlocB), c_int64_t)
  lenC = int(max(1, ldc_ll), c_int64_t) * int(max(0, nlocC), c_int64_t)

  ! Allocate GPU arrays (stay on GPU)
  call gpu_alloc_complex(lenA, dA)
  call gpu_alloc_complex(lenB, dB)
  call gpu_alloc_complex(lenC, dC)

  ! Fill A and B directly on GPU
  call fill_A_local_gpu(dA, lda_ll, n, mbi, nbi, nprow, npcol, myrow, mycol)
  call fill_B_local_gpu(dB, ldb_ll, n, mbi, nbi, nprow, npcol, myrow, mycol)

  ! Print sums (sum computed on GPU; only 1 complex copied to host)
  call device_sum_z(lenA, dA, sumA_h)
  call device_sum_z(lenB, dB, sumB_h)
  print *, 'rank', myrank, 'sum_locA =', sumA_h
  print *, 'rank', myrank, 'sum_locB =', sumB_h

  ! ---- SLATE matvec using GPU-resident distributed data ----
  ! C := A * B
  call slate_zgemm_from_devices(int(n, c_int64_t), int(mb, c_int64_t), int(nb, c_int64_t), &
                                nprow, npcol, dA, int(lda_ll, c_int64_t), &
                                dB, int(ldb_ll, c_int64_t), &
                                dC, int(ldc_ll, c_int64_t), MPI_Comm_c2f(MPI_COMM_WORLD))

  call device_sum_z(lenC, dC, sumC_h)
  print *, 'rank', myrank, 'sum_locC =', sumC_h

  call gpu_free(dC)
  call gpu_free(dB)
  call gpu_free(dA)

  call blacs_gridexit(ictxt)
  call MPI_Finalize(ierr)
end program demo_zmatvec

