#ifndef CONTRA_ROCM_BUILTINS_HPP
#define CONTRA_ROCM_BUILTINS_HPP

#include "config.h"

void __syncthreads();

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

__attribute__((const)) size_t GET_LOCAL_SIZE(uint dim)
{
    __constant byte_t * p = __builtin_amdgcn_dispatch_ptr();
    uint group_id, grid_size, group_size;

    switch(dim) {
    case 0:
        group_id = __builtin_amdgcn_workgroup_id_x();
        group_size = __builtin_amdgcn_workgroup_size_x();
        grid_size = *(__constant uint*)(p + 12); // love it!
        break;
    case 1:
        group_id = __builtin_amdgcn_workgroup_id_y();
        group_size = __builtin_amdgcn_workgroup_size_y();
        grid_size = *(__constant uint*)(p + 16); // love it!
        break;
    case 2:
        group_id = __builtin_amdgcn_workgroup_id_z();
        group_size = __builtin_amdgcn_workgroup_size_z();
        grid_size = *(__constant uint*)(p + 20); // love it!
        break;
    default:
        group_id = 0;
        grid_size = 0;
        group_size = 1;
        break;
    }
    uint r = grid_size - group_id * group_size;
    return (r < group_size) ? r : group_size;
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
