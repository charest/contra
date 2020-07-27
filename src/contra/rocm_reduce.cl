#include "config.h"

#include "librtrocm/rocm_scratch.h"
#include "librtrocm/rocm_builtins.h"

void init(volatile byte_t*);
void apply(volatile byte_t*, volatile byte_t*);
void fold(byte_t*, byte_t*, byte_t*);

//==============================================================================
/// Reduce accross blocks
//==============================================================================
void warpReduce(
    volatile byte_t *sdata,
    unsigned int tid,
    unsigned int blockSize,
    unsigned int data_size)
{
  if (blockSize >= 64) apply(sdata + data_size*tid, sdata + data_size*(tid + 32));
  if (blockSize >= 32) apply(sdata + data_size*tid, sdata + data_size*(tid + 16));
  if (blockSize >= 16) apply(sdata + data_size*tid, sdata + data_size*(tid +  8));
  if (blockSize >=  8) apply(sdata + data_size*tid, sdata + data_size*(tid +  4));
  if (blockSize >=  4) apply(sdata + data_size*tid, sdata + data_size*(tid +  2));
  if (blockSize >=  2) apply(sdata + data_size*tid, sdata + data_size*(tid +  1));
}

//==============================================================================
/// Main reduction kernel
//==============================================================================
void kernel reduce6(
  global byte_t * g_idata,
  global byte_t * g_odata,
  unsigned int data_size,
  unsigned int n,
  unsigned int blockSize
) {
  __local byte_t * sdata = GET_LOCAL_MEM_PTR();

  unsigned int gid = GET_GROUP_ID(0);
  unsigned int tid = GET_LOCAL_ID(0);
  unsigned int i = gid*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*GET_NUM_GROUPS(0);

  //sdata[tid] = 0;
  init( sdata + tid*data_size );
 
  while (i < n) {
    //sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    fold(
        g_idata + data_size*i,
        g_idata + data_size*(i+blockSize),
        sdata//   + data_size*tid
    );
    i += gridSize;
  }

  SYNC();

  if (blockSize >= 512) {
    if (tid < 256) 
      //sdata[tid] = P(sdata[tid], sdata[tid + 256]); }
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + 256)
      );
    SYNC();
  }
  if (blockSize >= 256) {
    if (tid < 128)
      //sdata[tid] = P(sdata[tid], sdata[tid + 128]);
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + 128)
      );
    SYNC();
  }
  if (blockSize >= 128) {
    if (tid < 64) 
      //sdata[tid] = P(sdata[tid], sdata[tid + 64]);
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + 64)
      );
    SYNC();
  }
  
  if (tid < 32) warpReduce(sdata, tid, blockSize, data_size);
  if (tid == 0) {
    //g_odata[blockIdx.x] = sdata[0];
    __builtin_memcpy(g_odata + data_size*gid, sdata, data_size);
  }
}
