#include "config.hpp"
#include <stdio.h>

#include "cuda_reduce.hpp"

#ifdef __CUDACC__

extern "C" {
//==============================================================================
/// Main reduction kernel
//==============================================================================
__global__
void reduce(
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
  unsigned int i = blockIdx.x*blockDim.x + tid;

  //sdata[tid] = 0;
  InitPtr( sdata + tid*data_size );

  if (i < n) {
    ApplyPtr(
      sdata + data_size*tid,
      g_idata + data_size*i
    );
  }
  
  __syncthreads();

  for(unsigned int s2=blockDim.x, s=s2/2; s>0; s>>=1) { 
    if (tid < s) {

      //sdata[tid] += sdata[tid + s];
      FoldPtr(
          sdata + data_size*tid,
          sdata + data_size*(tid + s)
      );

      // sdata[tid] += sdata[tid + (s2-1)];
      if (2*s < s2) {
        if (tid==0) {
          FoldPtr(
              sdata + data_size*tid,
              sdata + data_size*(tid + s2-1)
          );
        }
      }

    } // tid < s

    s2 = s;
    __syncthreads();
  } // for

  if (tid == 0) {
    //g_odata[blockIdx.x] = sdata[0];
    memcpy(g_odata + data_size*blockIdx.x, sdata, data_size);
  }
}

} // extern C

#endif //__CUDACC__
