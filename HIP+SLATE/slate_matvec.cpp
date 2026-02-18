#include <mpi.h>
#include <slate/slate.hh>
#include <complex>
#include <cstdint>

extern "C" void slate_zgemm_from_devices(
    int64_t n,
    int64_t mb,
    int64_t nb,
    int p, int q,
    void* dA, int64_t lda,
    void* dB, int64_t ldb,
    void* dC, int64_t ldc,
    int fcomm )
{
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)fcomm);

    using scalar_t = std::complex<double>;
    auto* A0 = reinterpret_cast<scalar_t*>(dA);
    auto* B0 = reinterpret_cast<scalar_t*>(dB);
    auto* C0 = reinterpret_cast<scalar_t*>(dC);

    // One GPU per rank:
    scalar_t* Aarray[1] = { A0 };
    scalar_t* Barray[1] = { B0 };
    scalar_t* Carray[1] = { C0 };

    // Map the *existing device memory* into SLATE matrices.
    auto A = slate::Matrix<scalar_t>::fromDevices(
        n, n, Aarray, /*num_devices=*/1, lda, mb, nb, p, q, comm);

    auto B = slate::Matrix<scalar_t>::fromDevices(
        n, 1, Barray, /*num_devices=*/1, ldb, mb, nb, p, q, comm);

    auto C = slate::Matrix<scalar_t>::fromDevices(
        n, 1, Carray, /*num_devices=*/1, ldc, mb, nb, p, q, comm);

    scalar_t alpha = scalar_t(1.0, 0.0);
    scalar_t beta  = scalar_t(0.0, 0.0);

    // Force GPU execution
    slate::Options opts;
    opts.insert({slate::Option::Target, slate::Target::Devices});

    // C := alpha*A*B + beta*C  (distributed, GPU-resident)
    slate::gemm(slate::Op::NoTrans, slate::Op::NoTrans,
                alpha, A, B, beta, C, opts);
}

