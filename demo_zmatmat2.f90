program demo_zmatmat
  use mpi
  use iso_c_binding, only: c_double_complex, c_long_long
  use slate_zmatmat_mod
  implicit none

  integer, parameter :: root = 0
  integer :: ierr, myrank, nprocs
  integer :: ictxt, nprow, npcol, myrow, mycol
  external :: blacs_get, blacs_gridinit, blacs_gridinfo, blacs_gridexit, blacs_exit
  integer :: numroc
  external :: numroc

  integer(c_long_long) :: n, mb, nb
  integer, parameter   :: mbi = 256, nbi = 256

  integer(c_long_long) :: lda_ll, ldb_ll, ldc_ll
  integer :: mlocA, nlocA, mlocB, nlocB, mlocC, nlocC
  integer :: lenA, lenB, lenC
  integer :: ii

  complex(c_double_complex), allocatable :: A_local(:), B_local(:), C_local(:)
  complex(c_double_complex), allocatable :: C_ref(:,:)   ! global reference on root
  real(8),               allocatable :: Cref_pack(:)     ! packed for Bcast

  double precision :: t_dev0, t_dev1, t_cpu0, t_cpu1
  double precision :: n2_diff_loc, n2_diff, n2_cref_loc, n2_cref
  double precision :: sum_loc, sum_tot

  ! ---------- MPI / BLACS ----------
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! choose n; keep moderate for mat-mat
  n  = 2048_c_long_long
  mb = mbi
  nb = nbi

  call choose_grid(nprocs, nprow, npcol)
  call blacs_get(-1, 0, ictxt)
  call blacs_gridinit(ictxt, 'Row', nprow, npcol)
  call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)

  ! local sizes
  mlocA = numroc(int(n,4), mbi, myrow, 0, nprow)
  nlocA = numroc(int(n,4), nbi, mycol, 0, npcol)

  mlocB = numroc(int(n,4), mbi, myrow, 0, nprow)
  nlocB = numroc(int(n,4), nbi, mycol, 0, npcol)

  mlocC = numroc(int(n,4), mbi, myrow, 0, nprow)
  nlocC = numroc(int(n,4), nbi, mycol, 0, npcol)

  lda_ll = max(1_c_long_long, int(mlocA, kind=c_long_long))
  ldb_ll = max(1_c_long_long, int(mlocB, kind=c_long_long))
  ldc_ll = max(1_c_long_long, int(mlocC, kind=c_long_long))

  lenA = max(1, int(lda_ll)) * max(0, nlocA)
  lenB = max(1, int(ldb_ll)) * max(0, nlocB)
  lenC = max(1, int(ldc_ll)) * max(0, nlocC)

  if (myrank == root) then
    print *, 'n =', n, '  mb =', mb, '  nb =', nb
    print *, 'nprow =', nprow, '  npcol =', npcol
  end if

  allocate(A_local(lenA), B_local(lenB), C_local(lenC))

  call fill_A_local_mat(A_local, int(lda_ll), int(n), mbi, nbi, nprow, npcol, myrow, mycol)
  call fill_B_local_mat(B_local, int(ldb_ll), int(n), mbi, nbi, nprow, npcol, myrow, mycol)
  if (size(C_local) > 0) C_local = (0.0d0, 0.0d0)

  ! ---------- GPU (SLATE) run: repeat a few times as in vec code ----------
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t_dev0 = MPI_Wtime()

  do ii = 0, 5
    if (myrank == root) print *, 'GPU iteration ', ii
    call slate_zmatmat(n, A_local, B_local, C_local)
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    B_local = C_local
  end do

  t_dev1 = MPI_Wtime()

  ! ---------- CPU reference on rank 0: C_ref = A * B (global, single task) ----------
  if (myrank == root) then
    allocate(C_ref(int(n), int(n)))
    t_cpu0 = MPI_Wtime()
    call cpu_reference_matmat(int(n), C_ref)
    t_cpu1 = MPI_Wtime()

    ! pack into real*8 buffer for broadcast
    allocate(Cref_pack(2*int(n)*int(n)))
    Cref_pack = transfer(C_ref, Cref_pack, 2*int(n)*int(n))
  end if

  ! local sum over my piece of C (just for a simple scalar)
  sum_loc = 0.0d0
  if (size(C_local) > 0) sum_loc = sum(C_local)

  ! ---------- Broadcast C_ref and compute local Frobenius error ----------
  if (myrank /= root) allocate(Cref_pack(2*int(n)*int(n)))
  call MPI_Bcast(Cref_pack, 2*int(n,4)*int(n,4), MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD, ierr)

  if (myrank /= root) allocate(C_ref(int(n),int(n)))
  C_ref = transfer(Cref_pack, C_ref, int(n)*int(n))
  if (allocated(Cref_pack)) deallocate(Cref_pack)

  call local_mat_error(C_local, int(ldc_ll), int(n), mbi, nbi, nprow, npcol, myrow, mycol, &
                       C_ref, n2_diff_loc, n2_cref_loc)

  call MPI_Allreduce(n2_diff_loc,  n2_diff,   1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(n2_cref_loc,  n2_cref,   1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(sum_loc,      sum_tot,   1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)

  ! ---------- Printing block analogous to demo_zmatvec.f90 ----------
  if (myrank == root) then
    write(*,'(A,I0,A,I0,A,I0,A,F8.3,A,F8.3,A,ES12.5)') &
      'n=', int(n), '  p=', nprow, '  q=', npcol, &
      '  GPU(s)=', t_dev1 - t_dev0, '  CPU(s)=', t_cpu1 - t_cpu0, &
      '  rel_err=', sqrt(n2_diff) / max(sqrt(n2_cref), 1d-300)
    call flush(6)

    if (n >= 3) then
      print *, 'C_ref(1,1:3) = ', C_ref(1,1:3)
    else
      print *, 'C_ref(1,1) = ', C_ref(1,1)
    end if
    if (lenC >= 3) then
      print *, 'First 3 local entries of C_local:', C_local(1:3)
    else if (lenC > 0) then
      print *, 'First local entry of C_local:', C_local(1)
    else
      print *, 'No local entries of C_local on root'
    end if
    print *, 'sum_tot (sum of C over all ranks): ', sum_tot
  end if

  call blacs_gridexit(ictxt)
  call MPI_Finalize(ierr)

contains

  ! ---------- Deterministic test data ----------
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

  ! ---------- Fill local tiles of A (n x n) ----------
  subroutine fill_A_local_mat(Aloc, lda_i, n_i, mb_i, nb_i, p, q, myr, myc)
    complex(c_double_complex), intent(out) :: Aloc(:)
    integer, intent(in) :: lda_i, n_i, mb_i, nb_i, p, q, myr, myc
    integer :: mlocA_i, nlocA_i
    integer :: lbr, lbc, l_i0, l_j0, h, w
    integer :: IBr, IBc, gi0, gj0, ii, jj, gi, gj, loc_idx

    mlocA_i = numroc(n_i, mb_i, myr, 0, p)
    nlocA_i = numroc(n_i, nb_i, myc, 0, q)
    if (mlocA_i == 0 .or. nlocA_i == 0) return

    lbr = 0
    do while (lbr*mb_i < mlocA_i)
      l_i0 = lbr*mb_i + 1
      h    = min(mb_i, mlocA_i - (l_i0-1))
      IBr  = myr + lbr*p
      gi0  = IBr*mb_i + 1

      lbc = 0
      do while (lbc*nb_i < nlocA_i)
        l_j0 = lbc*nb_i + 1
        w    = min(nb_i, nlocA_i - (l_j0-1))
        IBc  = myc + lbc*q
        gj0  = IBc*nb_i + 1

        do jj = 0, w-1
          do ii = 0, h-1
            gi = gi0 + ii
            gj = gj0 + jj
            if (gi <= n_i .and. gj <= n_i) then
              loc_idx = (l_i0 + ii) + (l_j0 + jj - 1)*lda_i
              Aloc(loc_idx) = valA(gi, gj, n_i)
            end if
          end do
        end do

        lbc = lbc + 1
      end do
      lbr = lbr + 1
    end do
  end subroutine fill_A_local_mat

  ! ---------- Fill local tiles of B (n x n) ----------
  subroutine fill_B_local_mat(Bloc, ldb_i, n_i, mb_i, nb_i, p, q, myr, myc)
    complex(c_double_complex), intent(out) :: Bloc(:)
    integer, intent(in) :: ldb_i, n_i, mb_i, nb_i, p, q, myr, myc
    integer :: mlocB_i, nlocB_i
    integer :: lbr, lbc, l_i0, l_j0, h, w
    integer :: IBr, IBc, gi0, gj0, ii, jj, gi, gj, loc_idx

    mlocB_i = numroc(n_i, mb_i, myr, 0, p)
    nlocB_i = numroc(n_i, nb_i, myc, 0, q)
    if (mlocB_i == 0 .or. nlocB_i == 0) return

    lbr = 0
    do while (lbr*mb_i < mlocB_i)
      l_i0 = lbr*mb_i + 1
      h    = min(mb_i, mlocB_i - (l_i0-1))
      IBr  = myr + lbr*p
      gi0  = IBr*mb_i + 1

      lbc = 0
      do while (lbc*nb_i < nlocB_i)
        l_j0 = lbc*nb_i + 1
        w    = min(nb_i, nlocB_i - (l_j0-1))
        IBc  = myc + lbc*q
        gj0  = IBc*nb_i + 1

        do jj = 0, w-1
          do ii = 0, h-1
            gi = gi0 + ii
            gj = gj0 + jj
            if (gi <= n_i .and. gj <= n_i) then
              loc_idx = (l_i0 + ii) + (l_j0 + jj - 1)*ldb_i
              Bloc(loc_idx) = valB(gi, gj, n_i)
            end if
          end do
        end do

        lbc = lbc + 1
      end do
      lbr = lbr + 1
    end do
  end subroutine fill_B_local_mat

  ! ---------- CPU reference GEMM on rank 0: C_ref = A * B ----------
  subroutine cpu_reference_matmat(n_i, Cref)
    integer, intent(in) :: n_i
    complex(c_double_complex), intent(out) :: Cref(:,:)
    integer :: i, j, k
    complex(c_double_complex) :: s

    if (size(Cref,1) /= n_i .or. size(Cref,2) /= n_i) stop "Cref wrong size"

    do j = 1, n_i
      do i = 1, n_i
        s = (0.0d0, 0.0d0)
        do k = 1, n_i
          s = s + valA(i, k, n_i) * valB(k, j, n_i)
        end do
        Cref(i, j) = s
      end do
    end do
  end subroutine cpu_reference_matmat

  ! ---------- Local Frobenius error for my part of C ----------
  subroutine local_mat_error(Cloc, ldc_i, n_i, mb_i, nb_i, p, q, myr, myc, Cref, n2diff_loc, n2ref_loc)
    complex(c_double_complex), intent(in) :: Cloc(:)
    integer, intent(in) :: ldc_i, n_i, mb_i, nb_i, p, q, myr, myc
    complex(c_double_complex), intent(in) :: Cref(:,:)
    double precision, intent(out) :: n2diff_loc, n2ref_loc

    integer :: mlocC_i, nlocC_i
    integer :: lbr, lbc, l_i0, l_j0, h, w
    integer :: IBr, IBc, gi0, gj0, ii, jj, gi, gj, loc_idx
    complex(c_double_complex) :: d

    n2diff_loc = 0.0d0
    n2ref_loc  = 0.0d0

    mlocC_i = numroc(n_i, mb_i, myr, 0, p)
    nlocC_i = numroc(n_i, nb_i, myc, 0, q)
    if (mlocC_i == 0 .or. nlocC_i == 0 .or. size(Cloc) == 0) return

    lbr = 0
    do while (lbr*mb_i < mlocC_i)
      l_i0 = lbr*mb_i + 1
      h    = min(mb_i, mlocC_i - (l_i0-1))
      IBr  = myr + lbr*p
      gi0  = IBr*mb_i + 1

      lbc = 0
      do while (lbc*nb_i < nlocC_i)
        l_j0 = lbc*nb_i + 1
        w    = min(nb_i, nlocC_i - (l_j0-1))
        IBc  = myc + lbc*q
        gj0  = IBc*nb_i + 1

        do jj = 0, w-1
          do ii = 0, h-1
            gi = gi0 + ii
            gj = gj0 + jj
            if (gi <= n_i .and. gj <= n_i) then
              loc_idx = (l_i0 + ii) + (l_j0 + jj - 1)*ldc_i
              d = Cloc(loc_idx) - Cref(gi, gj)
              n2diff_loc = n2diff_loc + dble(d*conjg(d))
              n2ref_loc  = n2ref_loc  + dble(Cref(gi,gj)*conjg(Cref(gi,gj)))
            end if
          end do
        end do

        lbc = lbc + 1
      end do
      lbr = lbr + 1
    end do
  end subroutine local_mat_error

end program demo_zmatmat

