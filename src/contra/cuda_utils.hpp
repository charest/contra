#ifndef CONTRA_CUDA_UTILS_HPP
#define CONTRA_CUDA_UTILS_HPP

#include "config.hpp"
#include <stdio.h>


typedef void(*init_t)(volatile byte_t*);
typedef void(*apply_t)(volatile byte_t*, volatile byte_t*);
typedef void(*fold_t)(byte_t*, byte_t*, byte_t*);

__device__ void Init(volatile byte_t* lhs) {
    *(int_t*)(lhs) = 0;
    *(real_t*)(lhs + sizeof(int_t)) = 0;
}

__device__ void Apply(volatile byte_t* lhs, volatile byte_t* rhs) {
    int_t * lhs_int = (int_t*)lhs;
    int_t * rhs_int = (int_t*)rhs;
    *lhs_int = (*lhs_int) + (*rhs_int);
    
    size_t int_size = sizeof(int_t);
    real_t * lhs_real = (real_t*)(lhs + int_size);
    real_t * rhs_real = (real_t*)(rhs + int_size);
    *lhs_real = (*lhs_real) + (*rhs_real);
}

__device__ void Fold (byte_t* lhs, byte_t* rhs, byte_t* res) {
    int_t * lhs_int = (int_t*)lhs;
    int_t * rhs_int = (int_t*)rhs;
    int_t * res_int = (int_t*)res;
    *res_int += (*lhs_int) + (*rhs_int);
    
    size_t int_size = sizeof(int_t);
    real_t * lhs_real = (real_t*)(lhs + int_size);
    real_t * rhs_real = (real_t*)(rhs + int_size);
    real_t * res_real = (real_t*)(res + int_size);
    *res_real += (*lhs_real) + (*rhs_real);
}

__device__ init_t  dev_InitPtr  = Init;
__device__ apply_t dev_ApplyPtr = Apply;
__device__ fold_t  dev_FoldPtr  = Fold;
  
  
  

//==============================================================================
/// Reduction
//==============================================================================
template <unsigned int blockSize>
__device__
void warpReduce(
    volatile byte_t *sdata,
    unsigned int tid,
    unsigned int data_size,
    apply_t ApplyPtr)
{
  if (blockSize >= 64) ApplyPtr(sdata + data_size*tid, sdata + data_size*(tid + 32));
  if (blockSize >= 32) ApplyPtr(sdata + data_size*tid, sdata + data_size*(tid + 16));
  if (blockSize >= 16) ApplyPtr(sdata + data_size*tid, sdata + data_size*(tid +  8));
  if (blockSize >=  8) ApplyPtr(sdata + data_size*tid, sdata + data_size*(tid +  4));
  if (blockSize >=  4) ApplyPtr(sdata + data_size*tid, sdata + data_size*(tid +  2));
  if (blockSize >=  2) ApplyPtr(sdata + data_size*tid, sdata + data_size*(tid +  1));
}

template <unsigned int blockSize>
__global__
void reduce6(
  byte_t * g_idata,
  byte_t * g_odata,
  unsigned int data_size,
  unsigned int n,
  init_t InitPtr,
  apply_t ApplyPtr,
  fold_t FoldPtr
) {

  extern __shared__ byte_t sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
 
  //sdata[tid] = 0;
  (*InitPtr)( sdata + tid*data_size );

  while (i < n) {
    //sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    FoldPtr(
        g_idata + data_size*i,
        g_idata + data_size*(i+blockSize),
        sdata   + data_size*tid
    );
    i += gridSize;
  }
   __syncthreads();

  if (blockSize >= 512) {
    if (tid < 256) 
      //sdata[tid] = P(sdata[tid], sdata[tid + 256]); }
      ApplyPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + 256)
      );
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128)
      //sdata[tid] = P(sdata[tid], sdata[tid + 128]);
      ApplyPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + 128)
      );
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) 
      //sdata[tid] = P(sdata[tid], sdata[tid + 64]);
      ApplyPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + 64)
      );
    __syncthreads();
  }

  if (tid < 32) warpReduce<blockSize>(sdata, tid, data_size, ApplyPtr);
  if (tid == 0) {
    //g_odata[blockIdx.x] = sdata[0];
    memcpy(g_odata + data_size*blockIdx.x, sdata, data_size);
  }
}

#endif // CONTRA_CUDA_UTILS_HPP
