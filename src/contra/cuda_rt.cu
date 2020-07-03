#include "config.hpp"
#include "cuda_rt.hpp"
#include "tasking_rt.hpp"

#include "cuda_utils.hpp"

#include <stdio.h>
#include <iostream>

extern "C" {

//==============================================================================
/// index space partitioning
//==============================================================================
__device__ void contra_cuda_index_space_create_from_partition(
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

//==============================================================================
/// Accessor read
//==============================================================================
__device__ void contra_cuda_accessor_read(
    contra_cuda_accessor_t * acc,
    void * data,
    int_t index)
{
  auto pos = (acc->offsets[threadIdx.x] + index) * acc->data_size;
  const byte_t * offset = static_cast<const byte_t*>(acc->data) + pos;
  memcpy(data, offset, acc->data_size);
  //printf("%d accessing %ld\n", threadIdx.x, acc->offsets[threadIdx.x] + index);
}

//==============================================================================
// prepare a reduction
//==============================================================================
__device__ void contra_cuda_set_reduction_value(
  void ** indata,
  void * data,
  size_t data_size)
{
  auto pos = data_size*threadIdx.x;
  byte_t * offset = static_cast<byte_t*>(*indata) + pos;
  memcpy(offset, data, data_size);
}

//==============================================================================
// temp reduction function
//==============================================================================
void contra_cuda_reduce(
  init_t dev_init,
  void ** indata,
  contra_index_space_t * is,
  void * outdata)
{
  size_t n = is->size();
  size_t data_size = sizeof(int_t) + sizeof(real_t);
 
#if 0
  void * dev_outdata;
  cudaMalloc(&dev_outdata, data_size); // num blocks
  size_t bytes = data_size * n;

  //init_t InitPtr;
  //cudaMemcpyFromSymbol(&InitPtr, dev_InitPtr, sizeof(init_t));

  //void * dev_i;
  //cudaMalloc(&dev_i, sizeof(init_t));
  //getInit<<<1,1>>>(dev_i);
  
  apply_t ApplyPtr;
  cudaMemcpyFromSymbol(&ApplyPtr, dev_ApplyPtr, sizeof(apply_t));

  fold_t FoldPtr;
  cudaMemcpyFromSymbol(&FoldPtr, dev_FoldPtr, sizeof(fold_t));

  reduce6<<<1,n,bytes>>>(
      (byte_t*)*indata,
      (byte_t*)dev_outdata,
      data_size,
      n,
      1,
      dev_init);//,
      //ApplyPtr,
      //FoldPtr); 
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "reduce6 failed with error \""
      << cudaGetErrorString(err) << "\"." << std::endl;
    abort();
  }

  cudaMemcpy(outdata, dev_outdata, data_size, cudaMemcpyDeviceToHost); 

  cudaFree(dev_outdata);
#endif
}

} // extern

