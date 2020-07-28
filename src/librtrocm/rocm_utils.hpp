#ifndef CONTRA_ROCM_UTILS_HPP
#define CONTRA_ROCM_UTILS_HPP

#include "config.hpp"

#include "hip/hip_runtime.h"

__global__
void rocm_init_kernel();

__global__
void rocm_copy_kernel(
    byte_t * src_data,
    byte_t * tgt_data,
    int_t * indices,
    size_t data_size,
    size_t size);

#endif // CONTRA_ROCM_UTILS_HPP
