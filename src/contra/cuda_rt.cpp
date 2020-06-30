#include "cuda_rt.hpp"
#include "tasking_rt.hpp"

#include "librt/dopevector.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {

/// global runtime for compiled cases
cuda_runtime_t CudaRuntime;

////////////////////////////////////////////////////////////////////////////////
/// Runtime definition
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Utility check function
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
void cuda_runtime_t::init(int dev_id) {
  auto err = cuInit(0);
  check(err);

  err = cuDeviceGet(&CuDevice, dev_id);
  check(err);
  printf( "Selected Device: %d\n", dev_id);

  err = cuCtxCreate(&CuContext, 0, CuDevice);
  check(err);

  CUjit_option options[6];
  void* values[6];
  options[0] = CU_JIT_WALL_TIME;
  values[0] = (void*)&walltime;
  options[1] = CU_JIT_INFO_LOG_BUFFER;
  values[1] = (void*)info_log;
  options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  values[2] = (void*)log_size;
  options[3] = CU_JIT_ERROR_LOG_BUFFER;
  values[3] = (void*)error_log;
  options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  values[4] = (void*)log_size;
  options[5] = CU_JIT_LOG_VERBOSE;
  values[5] = (void*)1;

  err = cuLinkCreate(6, options, values, &CuLinkState);
  check(err);

  IsStarted = true;
}

//==============================================================================
// shutdown runtime
//==============================================================================
void cuda_runtime_t::shutdown() {
  if (!IsStarted) return;

  auto err = cuLinkDestroy(CuLinkState);
  check(err);

  err = cuCtxDestroy(CuContext);
  check(err);

  IsStarted = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Public c interface
////////////////////////////////////////////////////////////////////////////////

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
    printf( "  Global memory:      %ld mb\n", props.totalGlobalMem / mb );
    printf( "  Shared memory:      %ld kb\n", props.sharedMemPerBlock / kb );
    printf( "  Constant memory:    %ld kb\n", props.totalConstMem / kb );
    printf( "  Block registers:    %d\n", props.regsPerBlock );
    printf( "  Warp size:          %d\n", props.warpSize );
    printf( "  Threads per block:  %d\n", props.maxThreadsPerBlock );
    printf( "  Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0],
        props.maxThreadsDim[1], props.maxThreadsDim[2] );
    printf( "  Max grid dimensions:  [%d, %d, %d]\n", props.maxGridSize[0],
        props.maxGridSize[1], props.maxGridSize[2] );
    printf( "\n" );
  }

  CudaRuntime.init(0);

  fflush(stdout);

}

//==============================================================================
// start runtime
//==============================================================================
void contra_cuda_shutdown() {
  CudaRuntime.shutdown();
}

//==============================================================================
// jit and register a kernel
//==============================================================================
void contra_cuda_register_kernel(const char * kernel)
{
  auto err = cuLinkAddData(
      CudaRuntime.CuLinkState,
      CU_JIT_INPUT_PTX,
      (void*)kernel,
      strlen(kernel) + 1,
      0, 0, 0, 0);
  check(err);

}

//==============================================================================
// launch a kernel
//==============================================================================
void contra_cuda_launch_kernel(
    char * name,
    contra_index_space_t * is,
    void *params[])
{
  void* cubin; 
  size_t cubin_size; 
  auto err = cuLinkComplete(CudaRuntime.CuLinkState, &cubin, &cubin_size);
  check(err);
  
  CUmodule CuModule;
  err = cuModuleLoadData(&CuModule, cubin);
  check(err);
  
  CUfunction CuFunction;
  err = cuModuleGetFunction(&CuFunction, CuModule, name);
  check(err);

  auto size = is->size();

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

  err = cuModuleUnload(CuModule);
  check(err);
}

//==============================================================================
// Copy array to device
//==============================================================================
void* contra_cuda_array2dev(dopevector_t * arr)
{
  void* dev_arr = nullptr;
  auto size = arr->bytes();
  auto err = cudaMalloc(&dev_arr, size);
  
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc fialed with error \""
      << cudaGetErrorString(err) << "\"." << std::endl;
    abort();
  }
  
  err = cudaMemcpy(dev_arr, arr->data, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed with error \""
      << cudaGetErrorString(err) << "\"." << std::endl;
    abort();
  }
  return dev_arr;
}

} // extern
