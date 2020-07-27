#ifndef CONTRA_ROCM_BUILTINS_HPP
#define CONTRA_ROCM_BUILTINS_HPP

#include "config.h"

// prototypes for internal functions
void * memcpy(void *, const void *, size_t);
void __memcpy_internal_aligned(void *, const void *, size_t, size_t);

inline uint GET_NUM_GROUPS(uint id) {
  __constant byte_t * p = (__constant byte_t*)__builtin_amdgcn_dispatch_ptr();
  uint n = *(__constant uint*)(p + 12); // love it!
  uint d = __builtin_amdgcn_workgroup_size_x();
  uint q = n / d;
  return q + (n > q*d);
}


// extracted from opencl/src/workgroup/wgbarrier.cl and opencl/src/misc/awif.cl
// of ROCm-Device-Libs

// LGKMC (LDS, GDS, Konstant, Message) is 4 bits
// EXPC (Export) is 3 bits
// VMC (VMem) is 4 bits
#define LGKMC_MAX 0xf                                                                             
#define EXPC_MAX 0x7
#define VMC_MAX 0xf
#define WAITCNT_IMM(LGKMC, EXPC, VMC) ((LGKMC << 8) | (EXPC << 4) | VMC)

#define SYNC() \
  __builtin_amdgcn_s_waitcnt(WAITCNT_IMM(0, EXPC_MAX, VMC_MAX)); \
  __builtin_amdgcn_s_barrier(); \
  __builtin_amdgcn_s_waitcnt(WAITCNT_IMM(0, EXPC_MAX, VMC_MAX));

#define GET_LOCAL_ID(id) \
  __builtin_amdgcn_workitem_id_x()

#define GET_GROUP_ID(id) \
  __builtin_amdgcn_workgroup_id_x()


#endif // CONTRA_ROCM_BUILTINS_HPP
