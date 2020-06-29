#include "cuda_rt.hpp"
#include "tasking_rt.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>

extern "C" {
  
CUmodule CuModule;
CUcontext CuContext;

//==============================================================================
// start runtime
//==============================================================================
void check(CUresult err){                                                                
  if(err != CUDA_SUCCESS){                                                                      
    const char* s;
    cuGetErrorString(err, &s);
    std::cerr << "CUDARuntime error: " << s << std::endl;
    abort();
  }                                                                                             
}                                                                                               


//==============================================================================
// start runtime
//==============================================================================
void contra_cuda_startup() {
  const int kb = 1024;
  const int mb = kb * kb;

  printf( "CUDA version:   v%d\n", CUDART_VERSION );

  int devCount;
  cudaGetDeviceCount(&devCount);
  printf( "CUDA Devices: \n\n" );

  for(int i = 0; i < devCount; ++i)
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    printf( "%d: %s: %d.%d\n", i, props.name, props.major, props.minor );
    printf( "  Global memory:      %d mb\n", props.totalGlobalMem / mb );
    printf( "  Shared memory:      %d kb\n", props.sharedMemPerBlock / kb );
    printf( "  Constant memory:    %d kb\n", props.totalConstMem / kb );
    printf( "  Block registers:    %d\n", props.regsPerBlock );
    printf( "  Warp size:          %d\n", props.warpSize );
    printf( "  Threads per block:  %d\n", props.maxThreadsPerBlock );
    printf( "  Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0],
        props.maxThreadsDim[1], props.maxThreadsDim[2] );
    printf( "  Max grid dimensions:  [%d, %d, %d]\n", props.maxGridSize[0],
        props.maxGridSize[1], props.maxGridSize[2] );
    printf( "\n" );
  }

  auto err = cuInit(0);
  check(err);

  CUdevice CuDevice;  
  err = cuDeviceGet(&CuDevice, 0);
  check(err);
  printf( "Selected Device: 0\n");

  err = cuCtxCreate(&CuContext, 0, CuDevice);
  check(err);

  fflush(stdout);

}

//==============================================================================
// jit and register a kernel
//==============================================================================
void contra_cuda_register_kernel(const char * kernel)
{
  auto err = cuModuleLoadData(&CuModule, (void*)kernel);
  check(err);
}

//==============================================================================
// launch a kernel
//==============================================================================
void contra_cuda_launch_kernel(char * name, contra_index_space_t * is)
{
  CUfunction CuFunction;
  auto err = cuModuleGetFunction(&CuFunction, CuModule, name);
  check(err);

  auto size = is->size();

  void *params[] = {};

  err = cuLaunchKernel(
      CuFunction,
      1, 1, 1,
      size, 1, 1,
      0,
      nullptr,
      params,
      nullptr);
  check(err);
 
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    std::cerr << "Kernel launch failed with error \""
      << cudaGetErrorString(cudaerr) << "\"." << std::endl;
    abort();
  }
}


} // extern
