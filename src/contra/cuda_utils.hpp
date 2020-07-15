#ifndef CONTRA_CUDA_UTILS_HPP
#define CONTRA_CUDA_UTILS_HPP

#include "config.hpp"

extern "C" {

void cuda_copy(
    byte_t * src_data,
    byte_t * tgt_data,
    int_t * indices,
    size_t data_size,
    size_t size,
    size_t num_threads,
    size_t num_blocks);

} // extern C

#endif // CONTRA_CUDA_UTILS_HPP
