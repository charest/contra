#include "config.h"

#include "librtrocm/rocm_scratch.h"
#include "librtrocm/rocm_builtins.h"

void init(volatile byte_t*);
void apply(volatile byte_t*, volatile byte_t*);

//==============================================================================
/// Main reduction kernel
//==============================================================================
void reduce(
  __global byte_t * g_idata,
  __global byte_t * g_odata,
  size_t data_size,
  int_t n
) {
  __local byte_t * sdata = GET_LOCAL_MEM_PTR();

  unsigned int tid = GET_LOCAL_ID(0);
  unsigned int gid = GET_GROUP_ID(0);
  unsigned int blockSize = GET_LOCAL_SIZE(0);
  unsigned int i = gid*blockSize + tid;

  //sdata[tid] = 0;
  init( sdata + tid*data_size );

  apply(
      sdata + data_size*tid,
      g_idata + data_size*i
  );
  
  SYNC();

  for(unsigned int s2=blockSize, s=s2/2; s>0; s>>=1) { 
    if (tid < s) {

      //sdata[tid] += sdata[tid + s];
      apply(
          sdata + data_size*tid,
          sdata + data_size*(tid + s)
      );

      // sdata[tid] += sdata[tid + (s2-1)];
      if (2*s < s2) {
        if (tid==0) {
          apply(
              sdata + data_size*tid,
              sdata + data_size*(tid + s2-1)
          );
        }
      }

    } // tid < s

    s2 = s;
    SYNC();
  } // for

  if (tid == 0) {
    //g_odata[blockIdx.x] = sdata[0];
    __builtin_memcpy(g_odata + data_size*gid, sdata, data_size);
  }
}
