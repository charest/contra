#include "config.hpp"
#include "rocm_utils.hpp"

#include "hip/hip_runtime.h"

////////////////////////////////////////////////////////////////////////////////
/// Init
////////////////////////////////////////////////////////////////////////////////
__global__
void rocm_init_kernel() {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy
////////////////////////////////////////////////////////////////////////////////
__global__
void rocm_copy_kernel(
    byte_t * src_data,
    byte_t * tgt_data,
    int_t * indices,
    size_t data_size,
    size_t size)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    auto src = src_data + indices[tid]*data_size;
    auto tgt = tgt_data + tid*data_size;
    memcpy(
      tgt,
      src,
      data_size );
  }
}
