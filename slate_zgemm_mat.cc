// slate_zgemm_mat.cc
#include <slate/slate.hh>
#include <mpi.h>
#include <complex>

extern "C"
void slate_zgemm_mat(
    int        fcomm,
    long long  n,
    void*      Aptr, long long lda,
    void*      Bptr, long long ldb,
    void*      Cptr, long long ldc,
    long long  mb,  long long nb,
    int        p,   int q,
    int        order_row )
{
    using z = std::complex<double>;

    MPI_Comm comm = MPI_Comm_f2c(fcomm);
    auto order = (order_row ? slate::GridOrder::Row
                            : slate::GridOrder::Col);

    z* A = static_cast<z*>(Aptr);
    z* B = static_cast<z*>(Bptr);
    z* C = static_cast<z*>(Cptr);

    // Global A(n,n), B(n,n), C(n,n) in ScaLAPACK layout.
    auto A_m = slate::Matrix<z>::fromScaLAPACK(
        n, n, A, lda, mb, nb, order, p, q, comm);
    auto B_m = slate::Matrix<z>::fromScaLAPACK(
        n, n, B, ldb, mb, nb, order, p, q, comm);
    auto C_m = slate::Matrix<z>::fromScaLAPACK(
        n, n, C, ldc, mb, nb, order, p, q, comm);

    // C = A * B  (alpha = 1, beta = 0), on GPUs.
    slate::gemm(z(1.0, 0.0), A_m, B_m, z(0.0, 0.0), C_m,
        slate::Options{ { slate::Option::Target, slate::Target::Devices } });
}

