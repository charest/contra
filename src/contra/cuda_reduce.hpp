#ifndef CONTRA_CUDA_REDUCE_HPP
#define CONTRA_CUDA_REDUCE_HPP

#include "config.hpp"
#include <stdio.h>

typedef void(*init_t)(volatile byte_t*);
typedef void(*apply_t)(volatile byte_t*, volatile byte_t*);
typedef void(*fold_t)(byte_t*, byte_t*, byte_t*);

#ifdef __CUDACC__

extern "C" {

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
);
} // extern C

#endif //__CUDACC__

#endif // CONTRA_CUDA_REDUCE_HPP
