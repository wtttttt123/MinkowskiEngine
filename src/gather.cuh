#ifndef GATHER_CUH
#define GATHER_CUH
#include "gpu.cuh"

#include <cusparse.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace minkowski {

template <typename th_int_type>
torch::Tensor weight_gather(torch::Tensor const &rows, torch::Tensor const &cols,
                    int64_t const dim_i,int64_t const dim_j, 
                    torch::Tensor const &mat_sum_weights,torch::Tensor const &mat_weights,
                    bool const is_sorted);

} // namespace minkowski
#endif