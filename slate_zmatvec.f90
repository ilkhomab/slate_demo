module slate_zmatvec_mod
  use, intrinsic :: iso_c_binding
  use mpi
  implicit none

  interface
    subroutine slate_zgemm_vec(fcomm, n, Aptr, lda, Bptr, ldb, Cptr, ldc, mb, nb, p, q, order_row) &
      bind(C, name="slate_zgemm_vec")
      use iso_c_binding
      integer(c_int),           value :: fcomm, p, q, order_row
      integer(c_long_long),     value :: n, lda, ldb, ldc, mb, nb
      type(c_ptr),              value :: Aptr, Bptr, Cptr
    end subroutine
  end interface

contains

  subroutine slate_zmatvec(n, A, B, C)
    use, intrinsic :: iso_c_binding
    implicit none
    ! --- Global size
    integer(c_long_long), intent(in) :: n
    ! --- Local tiles (must be TARGET for c_loc)
    complex(c_double_complex), intent(in),  target :: A(:)
    complex(c_double_complex), intent(in),  target :: B(:)
    complex(c_double_complex), intent(out), target :: C(:)

    ! --- MPI / BLACS
    logical :: init_flag
    integer :: ierr, myrank, nprocs
    integer :: ictxt, nprow, npcol, myrow, mycol
    external :: blacs_pinfo, blacs_get, blacs_gridinit, blacs_gridinfo, blacs_gridexit, blacs_exit
    integer :: numroc
    external :: numroc

    ! --- Block sizes and local dims
    integer(c_long_long) :: mb, nb, lda, ldb, ldc
    integer :: mlocA, nlocA, mlocB, nlocB, mlocC, nlocC
    integer :: mbi, nbi

    ! --- C interop
    type(c_ptr) :: Aptr, Bptr, Cptr
    integer(c_int) :: fcomm

    call MPI_Initialized(init_flag, ierr)           ! flag must be LOGICAL
    if (.not. init_flag) call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    call choose_grid(nprocs, nprow, npcol)
    call blacs_get(-1, 0, ictxt)
    call blacs_gridinit(ictxt, 'Row', nprow, npcol)
    call blacs_gridinfo(ictxt, nprow, npcol, myrow, mycol)

    ! Block sizes (tune)
    mb = 256_c_long_long;  nb = 256_c_long_long
    mbi = 256;             nbi = 256

    ! Local sizes under 2D block-cyclic
    mlocA = numroc(int(n,kind=4), mbi, myrow, 0, nprow)
    nlocA = numroc(int(n,kind=4), nbi, mycol, 0, npcol)
    mlocB = numroc(int(n,kind=4), mbi, myrow, 0, nprow)
    nlocB = numroc(1,              nbi, mycol, 0, npcol)
    mlocC = numroc(int(n,kind=4), mbi, myrow, 0, nprow)
    nlocC = numroc(1,              nbi, mycol, 0, npcol)

    lda = max(1_c_long_long, int(mlocA, kind=c_long_long))
    ldb = max(1_c_long_long, int(mlocB, kind=c_long_long))
    ldc = max(1_c_long_long, int(mlocC, kind=c_long_long))

    ! Sanity checks (avoid SIZE(...,kind=8) for portability)
    if (size(A) /= int(lda,kind=8) * int(nlocA,kind=8)) then
      write(*,*) 'ERROR: local A size mismatch. Expected ', int(lda,8)*int(nlocA,8), ' got ', int(size(A),8)
      error stop
    end if
    if (size(B) /= int(ldb,kind=8) * int(nlocB,kind=8)) then
      write(*,*) 'ERROR: local B size mismatch. Expected ', int(ldb,8)*int(nlocB,8), ' got ', int(size(B),8)
      error stop
    end if
    if (size(C) /= int(ldc,kind=8) * int(nlocC,kind=8)) then
      write(*,*) 'ERROR: local C size mismatch. Expected ', int(ldc,8)*int(nlocC,8), ' got ', int(size(C),8)
      error stop
    end if

    ! c_loc must not reference A(1) if size==0; MERGE would still evaluate both args.
    if (size(A) > 0) then
      Aptr = c_loc(A(1))
    else
      Aptr = c_null_ptr
    end if
    if (size(B) > 0) then
      Bptr = c_loc(B(1))
    else
      Bptr = c_null_ptr
    end if
    if (size(C) > 0) then
      Cptr = c_loc(C(1))
    else
      Cptr = c_null_ptr
    end if

    ! Pass the Fortran MPI handle directly (no MPI_Comm_c2f needed here)
    fcomm = int(MPI_COMM_WORLD, kind=c_int)

    call slate_zgemm_vec( fcomm, n, Aptr, lda, Bptr, ldb, Cptr, ldc, &
                          mb, nb, int(nprow,c_int), int(npcol,c_int), 1_c_int )

    call blacs_gridexit(ictxt)
    if (.not. init_flag) call MPI_Finalize(ierr)
  end subroutine slate_zmatvec

  subroutine choose_grid(nprocs, nprow, npcol)
    implicit none
    integer, intent(in)  :: nprocs
    integer, intent(out) :: nprow, npcol
    integer :: p
    p = int(sqrt(real(nprocs)))
    do while (mod(nprocs, p) /= 0); p = p - 1; end do
    nprow = p; npcol = nprocs / p
  end subroutine choose_grid

end module slate_zmatvec_mod

