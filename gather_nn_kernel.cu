#include "cuda.h"

/**
 * ref_dev        B,N,D
 * query_dev      B,M,D
 * ind_dev        B,N,K
 * out_feat_dev   B,N,K,D
 */
__global__ void gather_nn_kernel(
    float* out_feat_dev, float* ref_dev, float* query_dev, long* ind_dev,
    int64_t B, int64_t N, int64_t k, int64_t Dim, int64_t M
) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= B*N)
    return;
  unsigned int iB = index / N;
  unsigned int iN = index % N;

  long *ind_ptr = ind_dev + iB * N * k + iN * k;
  float *ref_ptr = ref_dev + iB * N * Dim;
  float *query_ptr = query_dev + iB * M * Dim;
  float *out_feat_ptr = out_feat_dev + iB * N * k * Dim + iN * k * Dim;
  for (int64_t i = 0; i < k; i++) {
    auto ind = *(ind_ptr + i);
    auto ref = ref_ptr + iN * Dim;
    auto query = query_ptr + ind * Dim;
    auto out_feat = out_feat_ptr + i * Dim;
    for (int64_t j = 0;  j < Dim; j++)
      out_feat[j] = query[j] - ref[j];
  }
  __syncthreads();
}

void gather_nn_dev(
  float* ref_dev, float* query_dev, long* ind_dev,
  int64_t B, int64_t N, int64_t k, int64_t Dim, int64_t M,
  float* out_feat_dev
) {
  int64_t all = B * N;
  dim3 blockSize;
  if (all <= 32)
    blockSize.x = 32;
  else if (all <= 64)
    blockSize.x = 64;
  else if (all <= 512)
    blockSize.x = 128;
  else if (all <= 1024)
    blockSize.x = 256;
  else
    blockSize.x = 512;
  dim3 gridSize((all + blockSize.x - 1) / blockSize.x);

  gather_nn_kernel<<<gridSize, blockSize>>>(
    out_feat_dev, ref_dev, query_dev, ind_dev,
    B, N, k, Dim, M
  );
  cudaDeviceSynchronize();
}
