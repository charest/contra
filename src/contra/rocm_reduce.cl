#include "config.h"

void init(volatile byte_t*);
void apply(volatile byte_t*, volatile byte_t*);
void fold(byte_t*, byte_t*, byte_t*);

void * memcpy(void *, const void *, size_t);

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
  local byte_t * sdata,
  unsigned int data_size,
  unsigned int n,
  unsigned int blockSize
) {

  unsigned int tid = get_local_id(0);
  unsigned int i = get_group_id(0)*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*get_num_groups(0);

  //sdata[tid] = 0;
  init( sdata + tid*data_size );

  while (i < n) {
    //sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    fold(
        g_idata + data_size*i,
        g_idata + data_size*(i+blockSize),
        sdata   + data_size*tid
    );
    i += gridSize;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (blockSize >= 512) {
    if (tid < 256) 
      //sdata[tid] = P(sdata[tid], sdata[tid + 256]); }
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + 256)
      );
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (blockSize >= 256) {
    if (tid < 128)
      //sdata[tid] = P(sdata[tid], sdata[tid + 128]);
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + 128)
      );
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (blockSize >= 128) {
    if (tid < 64) 
      //sdata[tid] = P(sdata[tid], sdata[tid + 64]);
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + 64)
      );
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid < 32) warpReduce(sdata, tid, blockSize, data_size);
  if (tid == 0) {
    //g_odata[blockIdx.x] = sdata[0];
    memcpy(g_odata + data_size*get_group_id(0), sdata, data_size);
  }
}
