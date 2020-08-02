#ifndef CONTRA_CUDA_REDUCE_HPP
#define CONTRA_CUDA_REDUCE_HPP

#include "config.hpp"
#include <stdio.h>

// fermi and kepler lack instructions to directly operate on shared
// memory.  So mark anything that might be shared as volatile.
typedef void(*init_t)(volatile byte_t*);
typedef void(*apply_t)(byte_t*, byte_t*, volatile byte_t*);
typedef void(*fold_t)(volatile byte_t*, volatile byte_t*);

#endif // CONTRA_CUDA_REDUCE_HPP
