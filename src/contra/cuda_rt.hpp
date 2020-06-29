#ifndef CONTRA_CUDA_RT_HPP
#define CONTRA_CUDA_RT_HPP

#include "config.hpp"

#include <cuda.h>

#include <map>
#include <string>

namespace contra {
  extern CUmodule CuModule;
  extern CUcontext CuContext;
}

extern "C" {

void contra_cuda_startup();
void contra_cuda_register_kernel(const char * kernel);

} // extern


#endif // LIBRT_LEGION_RT_HPP
