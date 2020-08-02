#include "config.hpp"
#include <stdio.h>

#include "cuda_reduce.hpp"

#ifdef __CUDACC__

extern "C" {
//==============================================================================
/// Reduce accross blocks
//==============================================================================
__device__
void warpReduce(
    volatile byte_t *sdata,
    unsigned int tid,
    unsigned int blockSize,
    unsigned int data_size,
    fold_t FoldPtr)
{
  // sdata[tid] += sdata[tid + 32];
  if (blockSize >= 64) FoldPtr(sdata + data_size*tid, sdata + data_size*(tid + 32));
  if (blockSize >= 32) FoldPtr(sdata + data_size*tid, sdata + data_size*(tid + 16));
  if (blockSize >= 16) FoldPtr(sdata + data_size*tid, sdata + data_size*(tid +  8));
  if (blockSize >=  8) FoldPtr(sdata + data_size*tid, sdata + data_size*(tid +  4));
  if (blockSize >=  4) FoldPtr(sdata + data_size*tid, sdata + data_size*(tid +  2));
  if (blockSize >=  2) FoldPtr(sdata + data_size*tid, sdata + data_size*(tid +  1));
}

//==============================================================================
/// Main reduction kernel
//==============================================================================
__global__
void reduce6(
  byte_t * g_idata,
  byte_t * g_odata,
  unsigned int data_size,
  unsigned int n,
  unsigned int blockSize,
  init_t InitPtr,
  apply_t ApplyPtr,
  fold_t FoldPtr
) {

  extern __shared__ byte_t sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;

  //sdata[tid] = 0;
  InitPtr( sdata + tid*data_size );

  while (i < n) {
    //sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    ApplyPtr(
        g_idata + data_size*i,
        g_idata + data_size*(i+blockSize),
        sdata   + data_size*tid
    );
    i += gridSize;
  }
   __syncthreads();

  if (blockSize >= 512) {
    if (tid < 256) 
      //sdata[tid] += sdata[tid + 256];
      FoldPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + 256)
      );
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128)
      //sdata[tid] += sdata[tid + 128];
      FoldPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + 128)
      );
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) 
      //sdata[tid] += sdata[tid + 64];
      FoldPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + 64)
      );
    __syncthreads();
  }

  if (tid < 32) warpReduce(sdata, tid, blockSize, data_size, FoldPtr);
  if (tid == 0) {
    //g_odata[blockIdx.x] = sdata[0];
    memcpy(g_odata + data_size*blockIdx.x, sdata, data_size);
  }
}

} // extern C

#endif //__CUDACC__
