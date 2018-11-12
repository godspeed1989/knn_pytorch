#include <torch/extension.h>

#define DEBUG 0

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gather_nn_dev(
  float* query_dev, long* ind_dev,
  int64_t B, int64_t N, int64_t k, int64_t Dim, int64_t M,
  float* out_feat_dev
);

/**
 * query_tensor   B,M,D
 * ind_tensor     B,N,K
 * feat_tensor    B,N,K,D
 */
int gather_nn(
  at::Tensor query_tensor,
  at::Tensor ind_tensor,
  at::Tensor out_feat_tensor) {

  CHECK_INPUT(query_tensor);
  CHECK_INPUT(ind_tensor);
  CHECK_INPUT(out_feat_tensor);

  AT_ASSERTM(ind_tensor.size(0) == query_tensor.size(0), "batch size not match 1")
  AT_ASSERTM(query_tensor.size(0) == out_feat_tensor.size(0), "batch size not match 2")
  AT_ASSERTM(ind_tensor.size(1) == out_feat_tensor.size(1), "N not match")
  AT_ASSERTM(ind_tensor.size(2) == out_feat_tensor.size(2), "k not match")
  AT_ASSERTM(query_tensor.size(2) == out_feat_tensor.size(3), "dim size not match")

  int64_t batch = query_tensor.size(0);
  int64_t M = query_tensor.size(1);
  int64_t Dim = query_tensor.size(2);
  int64_t k = ind_tensor.size(2);
  int64_t N = ind_tensor.size(1);
#if DEBUG
  std::cout << std::endl;
  std::cout << "B:" << batch << std::endl;
  std::cout << "M:" << M << std::endl;
  std::cout << "Dim:" << Dim << std::endl;
  std::cout << "k:" << k << std::endl;
  std::cout << "N:" << N << std::endl;
#endif
  float *query_dev = query_tensor.data<float>();
  long *ind_dev = ind_tensor.data<long>();
  float *out_feat_dev = out_feat_tensor.data<float>();

  gather_nn_dev(
    query_dev, ind_dev,
    batch, N, k, Dim, M,
    out_feat_dev
  );

  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_nn", &gather_nn, "gather k-nearest neighbors");
}
