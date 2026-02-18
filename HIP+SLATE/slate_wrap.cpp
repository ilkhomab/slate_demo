#include <mpi.h>
#include <complex>
#include <cstdint>
#include <slate/slate.hh>

extern "C" void slate_zgemm_from_devices(
    long long n,
    long long mb,
    long long nb,
    int p, int q,
    void* dA, long long lda,
    void* dB, long long ldb,
    void* dC, long long ldc,
    int fcomm )
{
  MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)fcomm);

  using scalar_t = std::complex<double>;
  auto* A0 = reinterpret_cast<scalar_t*>(dA);
  auto* B0 = reinterpret_cast<scalar_t*>(dB);
  auto* C0 = reinterpret_cast<scalar_t*>(dC);

  // One GPU per rank (masked by ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)
  scalar_t* Aarray[1] = { A0 };
  scalar_t* Barray[1] = { B0 };
  scalar_t* Carray[1] = { C0 };

  auto A = slate::Matrix<scalar_t>::fromDevices(
      (int64_t)n, (int64_t)n, Aarray, 1, (int64_t)lda, (int64_t)mb, (int64_t)nb, p, q, comm);

  auto B = slate::Matrix<scalar_t>::fromDevices(
      (int64_t)n, (int64_t)1, Barray, 1, (int64_t)ldb, (int64_t)mb, (int64_t)nb, p, q, comm);

  auto C = slate::Matrix<scalar_t>::fromDevices(
      (int64_t)n, (int64_t)1, Carray, 1, (int64_t)ldc, (int64_t)mb, (int64_t)nb, p, q, comm);

  scalar_t alpha(1.0, 0.0);
  scalar_t beta (0.0, 0.0);

  slate::Options opts;
  opts.insert({ slate::Option::Target, slate::Target::Devices });

  slate::gemm(alpha, A, B, beta, C, opts); 
}

