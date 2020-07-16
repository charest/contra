#include "config.hpp"

#include <stdio.h>

extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Copy
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Main copy kernel
//==============================================================================
__global__
void cuda_copy_kernel(
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

//==============================================================================
// Main exposed copy function
//==============================================================================
void cuda_copy(
    byte_t * src_data,
    byte_t * tgt_data,
    int_t * indices,
    size_t data_size,
    size_t size,
    size_t num_threads,
    size_t num_blocks)
{
  cuda_copy_kernel<<<num_threads, num_blocks>>>(
    src_data,
    tgt_data,
    indices,
    data_size,
    size);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Main init kernel
//==============================================================================
__global__
void cuda_init_kernel(
    byte_t * data,
    byte_t * init,
    size_t data_size,
    size_t size)
{
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    auto tgt = data + tid*data_size;
    memcpy(
      tgt,
      init,
      data_size );
  }
}

//==============================================================================
// Main exposed init function
//==============================================================================
void cuda_init(
    byte_t * data,
    byte_t * init,
    size_t data_size,
    size_t size,
    size_t num_threads,
    size_t num_blocks)
{
  cuda_init_kernel<<<num_threads, num_blocks>>>(
    data,
    init,
    data_size,
    size);
}

} // extern C

