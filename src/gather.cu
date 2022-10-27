#include "gpu.cuh"
#include "math_functions.cuh"

#include <cusparse.h>

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <torch/script.h>
#include "spmm.cuh"

namespace minkowski {

template <typename th_int_type>
torch::Tensor weight_gather(torch::Tensor const &rows, torch::Tensor const &cols,
                        int64_t const dim_i,int64_t const dim_j, 
                        torch::Tensor const &mat_sum_weights,torch::Tensor const &mat_weights, bool const is_sorted) {
#if defined __HIP_PLATFORM_HCC__
  TORCH_CHECK(false, "gather is not supported on HIP");
#elif defined(_WIN32) || defined(_WIN64)
  TORCH_CHECK(false, "gather CUDA is not supported on Windows");
#elif !defined(CUDART_VERSION)
  TORCH_CHECK(false, "CUDART_VERSION not defined");
#endif

  constexpr bool is_int32 = std::is_same<th_int_type, int32_t>::value;
  constexpr bool is_int64 = std::is_same<th_int_type, int64_t>::value;

  at::ScalarType int_scalar_type = std::is_same<th_int_type, int32_t>::value
                                       ? at::ScalarType::Int
                                       : at::ScalarType::Long;

  ASSERT(rows.scalar_type() == int_scalar_type, "int type mismatch.");

  ASSERT(rows.scalar_type() == cols.scalar_type(),"rows and cols must have the same scalar type.");
  ASSERT(rows.is_contiguous(), "rows must be contiguous");
  ASSERT(cols.is_contiguous(), "cols must be contiguous");

  ASSERT(rows.is_cuda(), "rows must be CUDA, but got CPU");
  ASSERT(cols.is_cuda(), "cols must be CUDA, but got CPU");
  ASSERT(mat_sum_weights.is_cuda(), "mat_sum_weights must be CUDA, but got CPU");
  ASSERT(at::cuda::check_device({rows, cols, mat_sum_weights}),
         "All inputs must be on the same device.");

  ASSERT(mat_sum_weights.dim() == 2, "Tensor 'mat_sum_weights' must have 2 dims, but has ",
         mat_sum_weights.dim());

  // int64_t dim_i = self.size(0);
  // int64_t dim_j = self.size(1);
  int64_t dim_k = mat_sum_weights.size(1);
  int64_t sizeY = mat_sum_weights.size(0);

  // Create tensors to view just the current set of matrices
  int64_t const nnz = rows.numel();

  LOG_DEBUG("Weight_Gather with dim_i:", dim_i, ", dim_j:", dim_j,", mat_sum_weights.size(0):", mat_sum_weights.size(0), ", mat_sum_weights.size(1):", mat_sum_weights.int64_t); 
  LOG_DEBUG("Creating a result mat shaped (", dim_k, ",", nnz, ")");
  torch::Tensor thrust_gather_output=at::zeros({nnz,1}, mat_sum_weights.options());
  torch::Tensor thrust_divide_output=at::zeros({nnz,1}, mat_weights.options());


  if ((dim_j == 0) || (dim_k == 0) || (nnz == 0)) {
    LOG_DEBUG("Skipping matmul dim_j:", dim_j, "dim_k:", dim_k, "nnz:", nnz);
    return thrust_divide_output;
  }

  // Dense matrices have to be contiguous for gather to work
  torch::Tensor const mat_sum_weights_contig = mat_sum_weights.contiguous();
  torch::Tensor const mat_weights_contig = mat_weights.contiguous();
  // Issue 308
  // auto cusparse_handle = at::cuda::getCurrentCUDASparseHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  cusparseHandle_t cusparse_handle = getCurrentCUDASparseHandle();
  cusparseSetStream(cusparse_handle, stream);

  cudaDataType cuda_data_type = minkowski::getTensorCudaDataType(mat_sum_weights_contig);
  th_int_type *row_indices_ptr =
      reinterpret_cast<th_int_type *>(rows.data_ptr());
  th_int_type *col_indices_ptr =
      reinterpret_cast<th_int_type *>(cols.data_ptr());
  
  // Iterate through each set of 2D matrices within the 3D
  // tensor inputs, performing a matrix multiply with each
  AT_DISPATCH_FLOATING_TYPES(mat_sum_weights.scalar_type(), "weight_gather", [&] {

    scalar_t *mat_sum_weights_ptr = reinterpret_cast<scalar_t *>(mat_sum_weights_contig.data_ptr());
    scalar_t *mat_weights_ptr = reinterpret_cast<scalar_t *>(mat_weights_contig.data_ptr());

    th_int_type *sorted_row_ptr, *sorted_col_ptr;
    //////////////////////////////////////
    // Sort the sparse matrix COO
    LOG_DEBUG("Is sorted", is_sorted);
    if (!is_sorted) {
      sorted_row_ptr =
          (th_int_type *)c10::cuda::CUDACachingAllocator::raw_alloc(
              2 * nnz * sizeof(th_int_type));
      sorted_col_ptr = sorted_row_ptr + nnz;
      LOG_DEBUG("Allocated sorted row col", nnz);

      // Copy the indices
      CUDA_CHECK(cudaMemcpy(sorted_row_ptr, row_indices_ptr,
                            nnz * sizeof(th_int_type),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(sorted_col_ptr, col_indices_ptr,
                            nnz * sizeof(th_int_type),
                            cudaMemcpyDeviceToDevice));

      THRUST_CHECK(thrust::sort_by_key(thrust::device,            //
                                       sorted_row_ptr,            // key begin
                                       sorted_row_ptr + nnz,      // key end
                                       thrust::make_zip_iterator( // value begin
                                           thrust::make_tuple(    //
                                               sorted_col_ptr,
                                               sorted_col_ptr    //
                                               )                  //
                                           )));
      LOG_DEBUG("sorted row", cudaDeviceSynchronize());
    } else {
      sorted_row_ptr = row_indices_ptr;
      sorted_col_ptr = col_indices_ptr;
      LOG_DEBUG("Initialized ptrs from inputs");
    }

    //////////////////////////////////////
    // scalar_t *new_gather_ptr, *new_sum_weights_ptr;
    // new_gather_ptr = (scalar_t *)c10::cuda::CUDACachingAllocator::raw_alloc(
    //       nnz * sizeof(scalar_t));
    // new_sum_weights_ptr = (scalar_t *)c10::cuda::CUDACachingAllocator::raw_alloc(
    //       nnz * sizeof(scalar_t));
    // CUDA_CHECK(cudaMemcpy(sorted_val_ptr, mat_sum_weights_ptr, sizeY * sizeof(scalar_t),
    //                         cudaMemcpyDeviceToDevice));
    //THRUST_CHECK(thrust::copy(sorted_val_ptr, sorted_val_ptr+sizeY,std::ostream_iterator<double>(std::cout,",,,")));
    
    THRUST_CHECK(thrust::gather(thrust::device,            //
                                sorted_row_ptr,            // key begin
                                sorted_row_ptr + nnz,      // key end
                                mat_sum_weights_ptr,
                                thrust_gather_output.data<scalar_t>()
                                ));
    
    THRUST_CHECK(thrust::transform(thrust::device,       //
                                mat_weights_ptr,            // key begin
                                mat_weights_ptr + nnz,      // key end
                                thrust_gather_output.data<scalar_t>(),
                                thrust_divide_output.data<scalar_t>(),
                                thrust::divides<scalar_t>())
                                );


    size_t workspace_buffer_size = 0;
    void *workspace_buffer = nullptr;

    if (!is_sorted) {
      LOG_DEBUG("Dealloc");
      c10::cuda::CUDACachingAllocator::raw_delete((void *)sorted_row_ptr);
    }

    if (workspace_buffer != nullptr) {
      cudaFree(workspace_buffer);
    }
    LOG_DEBUG("Dealloc finished", cudaDeviceSynchronize());
  });

  CUDA_CHECK(cudaDeviceSynchronize());

  return thrust_divide_output;
}

template torch::Tensor
weight_gather<int32_t>(torch::Tensor const &rows, torch::Tensor const &cols,
                   int64_t const dim_i,
                  int64_t const dim_j, torch::Tensor const &mat_sum_weights,torch::Tensor const &mat_weights,
                   bool const is_sorted);

} // namespace minkowski
