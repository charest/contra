#ifndef CONTRA_CUDA_RT_HPP
#define CONTRA_CUDA_RT_HPP

#include "config.hpp"

#include <cuda.h>

extern "C" {

//==============================================================================
// Runtime definition
//==============================================================================
struct cuda_runtime_t {
  CUdevice CuDevice;  
  CUcontext CuContext;
  CUlinkState CuLinkState;
  bool IsStarted = false;
 
  static constexpr auto log_size = 8192;
  float walltime = 0;
  char error_log[log_size];
  char info_log[8192];

  void init(int);
  void shutdown();

};

//==============================================================================
// public functions
//==============================================================================

void contra_cuda_startup();
void contra_cuda_shutdown();
void contra_cuda_register_kernel(const char * kernel);

} // extern


#endif // LIBRT_LEGION_RT_HPP
