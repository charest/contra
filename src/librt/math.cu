#include "config.hpp"

extern "C" {

//==============================================================================
/// index space partitioning
//==============================================================================
__device__ real_t fabs(real_t a) { return 
    int_t i,
    contra_cuda_partition_t * part,
    contra_index_space_t * is
    )
{ is->setup(part->offsets[i], part->offsets[i+1]); }

//==============================================================================
/// Accessor write
//==============================================================================
__device__ void contra_cuda_accessor_write(
    contra_cuda_accessor_t * acc,
    const void * data,
    int_t index)
{
  auto pos = (acc->offsets[threadIdx.x] + index) * acc->data_size;
  byte_t * offset = static_cast<byte_t*>(acc->data) + pos;
  memcpy(offset, data, acc->data_size);
}

} // extern

