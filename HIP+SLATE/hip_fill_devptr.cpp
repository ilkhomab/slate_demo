#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void hip_check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(e));
    std::abort();
  }
}

__device__ __forceinline__ hipDoubleComplex make_z(double re, double im) {
  hipDoubleComplex z; z.x = re; z.y = im; return z;
}

__device__ __forceinline__ hipDoubleComplex valA_dev(int gi, int gj) {
  int r1 = (3*gi + 5*gj) % 1024;
  int r2 = (7*gi - 11*gj) % 1024;   // keep negative values
  return make_z(double(r1)/1024.0, double(r2)/1024.0);
}

__device__ __forceinline__ hipDoubleComplex valB_dev(int gi) {
  int r1 = (13*gi) % 1024;
  int r2 = (17*gi) % 1024;
  return make_z(double(r1)/1024.0, -double(r2)/1024.0);
}

__global__ void fill_A_kernel(hipDoubleComplex* A, int lda, int mloc, int nloc,
                              int n, int mb, int nb, int p, int q, int myr, int myc)
{
  int li = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int lj = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
  if (li >= mloc || lj >= nloc) return;

  int gi = (((li) / mb) * p + myr) * mb + (li % mb) + 1;
  int gj = (((lj) / nb) * q + myc) * nb + (lj % nb) + 1;

  A[li + lj * lda] = (gi <= n && gj <= n) ? valA_dev(gi, gj) : make_z(0.0, 0.0);
}

__global__ void fill_B_kernel(hipDoubleComplex* V, int ldv, int mloc, int nloc,
                              int n, int mb, int nb, int p, int q, int myr, int myc)
{
  int li = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int lj = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
  if (li >= mloc || lj >= nloc) return;

  int gi = (((li) / mb) * p + myr) * mb + (li % mb) + 1;
  V[li + lj * ldv] = (gi <= n) ? valB_dev(gi) : make_z(0.0, 0.0);
}

__global__ void sum_kernel(const hipDoubleComplex* in, hipDoubleComplex* out, size_t n)
{
  size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  if (i >= n) return;

  atomicAdd(&out->x, in[i].x);
  atomicAdd(&out->y, in[i].y);
}

extern "C" void* hip_alloc_bytes(size_t nbytes) {
  void* p = nullptr;
  hip_check(hipMalloc(&p, nbytes), "hipMalloc");
  return p;
}

extern "C" void hip_free(void* p) {
  if (p) hip_check(hipFree(p), "hipFree");
}

static void set_rank_device_safely() {
  hip_check(hipSetDevice(0), "hipSetDevice(0)");
}

extern "C" void fill_A_local_hip_dev(void* Adev, int lda, int mloc, int nloc,
                                     int n, int mb, int nb, int p, int q, int myr, int myc)
{
  if (mloc <= 0 || nloc <= 0) return;

  set_rank_device_safely();

  dim3 block(16, 16, 1);
  dim3 grid((mloc + block.x - 1) / block.x,
            (nloc + block.y - 1) / block.y,
            1);

  hipLaunchKernelGGL(fill_A_kernel, grid, block, 0, 0,
                     (hipDoubleComplex*)Adev, lda, mloc, nloc, n, mb, nb, p, q, myr, myc);
  hip_check(hipGetLastError(), "launch fill_A_kernel");
  hip_check(hipDeviceSynchronize(), "sync fill_A_kernel");
}

extern "C" void fill_B_local_hip_dev(void* Bdev, int ldv, int mloc, int nloc,
                                     int n, int mb, int nb, int p, int q, int myr, int myc)
{
  if (mloc <= 0 || nloc <= 0) return;

  set_rank_device_safely();

  dim3 block(16, 16, 1);
  dim3 grid((mloc + block.x - 1) / block.x,
            (nloc + block.y - 1) / block.y,
            1);

  hipLaunchKernelGGL(fill_B_kernel, grid, block, 0, 0,
                     (hipDoubleComplex*)Bdev, ldv, mloc, nloc, n, mb, nb, p, q, myr, myc);
  hip_check(hipGetLastError(), "launch fill_B_kernel");
  hip_check(hipDeviceSynchronize(), "sync fill_B_kernel");
}

extern "C" void gpu_sum_complex(void* devptr, size_t n, hipDoubleComplex* sum_out)
{
  if (n == 0) {
    sum_out->x = 0.0;
    sum_out->y = 0.0;
    return;
  }

  set_rank_device_safely();

  hipDoubleComplex* d_in  = (hipDoubleComplex*)devptr;
  hipDoubleComplex* d_sum = nullptr;
  hip_check(hipMalloc(&d_sum, sizeof(hipDoubleComplex)), "hipMalloc d_sum");

  hipDoubleComplex zero{0.0, 0.0};
  hip_check(hipMemcpy(d_sum, &zero, sizeof(zero), hipMemcpyHostToDevice), "init d_sum");

  const int block = 256;
  const int grid  = int((n + block - 1) / block);

  hipLaunchKernelGGL(sum_kernel, dim3(grid), dim3(block), 0, 0, d_in, d_sum, n);
  hip_check(hipGetLastError(), "launch sum_kernel");
  hip_check(hipDeviceSynchronize(), "sync sum_kernel");

  hip_check(hipMemcpy(sum_out, d_sum, sizeof(*sum_out), hipMemcpyDeviceToHost), "copy sum D2H");
  hip_check(hipFree(d_sum), "hipFree d_sum");
}

extern "C" void gpu_get_first2_complex(void* devptr, size_t n, hipDoubleComplex out2[2])
{
  // out2 is host memory provided by Fortran
  out2[0].x = 0.0; out2[0].y = 0.0;
  out2[1].x = 0.0; out2[1].y = 0.0;

  if (n == 0 || devptr == nullptr) return;

  set_rank_device_safely();

  size_t k = (n >= 2) ? 2 : 1;
  hip_check(hipMemcpy(out2, devptr, k * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost),
            "gpu_get_first2_complex D2H");
}

extern "C" void print_gpu_info(int rank) {
  set_rank_device_safely();

  const char* hvd = std::getenv("HIP_VISIBLE_DEVICES");
  const char* rvd = std::getenv("ROCR_VISIBLE_DEVICES");
  const char* hsa = std::getenv("HSA_VISIBLE_DEVICES");

  int dev = -1;
  hip_check(hipGetDevice(&dev), "hipGetDevice");

  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, dev), "hipGetDeviceProperties");

  char pciBusId[64];
  pciBusId[0] = '\0';
  hipError_t e = hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), dev);
  if (e != hipSuccess) std::strncpy(pciBusId, "unknown", sizeof(pciBusId));

  std::fprintf(stdout,
    "[rank %d] hipDevice=%d name='%s' PCI='%s' HIP_VISIBLE_DEVICES='%s' ROCR_VISIBLE_DEVICES='%s' HSA_VISIBLE_DEVICES='%s'\n",
    rank, dev, prop.name, pciBusId,
    hvd ? hvd : "(unset)",
    rvd ? rvd : "(unset)",
    hsa ? hsa : "(unset)"
  );
  std::fflush(stdout);
}

