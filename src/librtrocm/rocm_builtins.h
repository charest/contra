#ifndef CONTRA_ROCM_BUILTINS_HPP
#define CONTRA_ROCM_BUILTINS_HPP

#include "config.h"

// prototypes for internal functions
void * memcpy(void *, const void *, size_t);
void __memcpy_internal_aligned(void *, const void *, size_t, size_t);

__attribute__((const)) size_t GET_NUM_GROUPS(uint dim) {
    __constant byte_t * p = (__constant byte_t*)__builtin_amdgcn_dispatch_ptr();

    uint n, d;
    switch(dim) {
    case 0:
        n = *(__constant uint*)(p + 12); // love it!
        d = __builtin_amdgcn_workgroup_size_x();
        break;
    case 1:
        n = *(__constant uint*)(p + 16); // love it!
        d = __builtin_amdgcn_workgroup_size_y();
        break;
    case 2:
        n = *(__constant uint*)(p + 20); // love it!
        d = __builtin_amdgcn_workgroup_size_z();
        break;
    default:
        n = 1;
        d = 1;
        break;
    }

    uint q = n / d;

    return q + (n > q*d);
}

__attribute__((const)) size_t GET_LOCAL_ID(uint dim)
{
    switch(dim) {
    case 0:
        return __builtin_amdgcn_workitem_id_x();
    case 1:
        return __builtin_amdgcn_workitem_id_y();
    case 2:
        return __builtin_amdgcn_workitem_id_z();
    default:
        return 0;
    }
}

__attribute__((const))  size_t GET_GROUP_ID(uint dim)
{
    switch(dim) {
    case 0:
        return __builtin_amdgcn_workgroup_id_x();
    case 1:
        return __builtin_amdgcn_workgroup_id_y();
    case 2:
        return __builtin_amdgcn_workgroup_id_z();
    default:
        return 0;
    }
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


#endif // CONTRA_ROCM_BUILTINS_HPP
