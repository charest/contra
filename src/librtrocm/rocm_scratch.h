#ifndef CONTRA_ROCM_SCRATCH_HPP
#define CONTRA_ROCM_SCRATCH_HPP

#include "config.h"

__attribute__((const)) __local ulong *__get_scratch_lds(void);

#define GET_LOCAL_MEM_PTR() \
  (__local byte_t *)__get_scratch_lds()

#endif // CONTRA_ROCM_SCRATCH_HPP
