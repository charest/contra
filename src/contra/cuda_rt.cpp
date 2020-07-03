#include "cuda_rt.hpp"
#include "cuda_utils.hpp"
#include "tasking_rt.hpp"

#include "librt/dopevector.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

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

void check(cudaError_t err, const char * msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " failed with error \""
      << cudaGetErrorString(err) << "\"." << std::endl;
    abort();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Runtime definition
////////////////////////////////////////////////////////////////////////////////

extern "C" {

/// global runtime for compiled cases
cuda_runtime_t CudaRuntime;

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
  
  IsStarted = true;
}

//==============================================================================
// shutdown runtime
//==============================================================================
void cuda_runtime_t::shutdown() {
  if (!IsStarted) return;

  auto err = cuCtxDestroy(CuContext);
  check(err);

  IsStarted = false;
}
  
  
void cuda_runtime_t::link(CUmodule &CuModule) {
  
  //------------------------------------
  // Start linker
  
  static constexpr auto log_size = 8192;
  float walltime = 0;
  char error_log[log_size];
  char info_log[8192];

  
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

  CUlinkState CuLinkState;
  auto err = cuLinkCreate(6, options, values, &CuLinkState);
  check(err);
  
  //------------------------------------
  // Populate linker

  // link library
  err = cuLinkAddFile(
      CuLinkState,
      CU_JIT_INPUT_LIBRARY,
      CONTRA_CUDA_LIBRARY,
      0, 0, 0);
  check(err);

  // link any ptx
  for (auto & ptx : Ptxs) {
    err = cuLinkAddData(
        CuLinkState,
        CU_JIT_INPUT_PTX,
        (void*)ptx.c_str(),
        ptx.size()+1,
        0, 0, 0, 0);
    check(err);
  }


  //------------------------------------
  // finish link
  void* cubin; 
  size_t cubin_size; 
  err = cuLinkComplete(CuLinkState, &cubin, &cubin_size);
  check(err);

  err = cuModuleLoadData(&CuModule, cubin);
  check(err);
  
  //------------------------------------
  // Destroy linker
  err = cuLinkDestroy(CuLinkState);
  check(err);
  

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
  CudaRuntime.Ptxs.emplace_back(kernel);
}

//==============================================================================
// launch a kernel
//==============================================================================
void contra_cuda_launch_kernel(
    char * name,
    contra_index_space_t * is,
    void *params[])
{

  KernelData* Kernel;

  auto it = CudaRuntime.Kernels.find(name);
  if (it != CudaRuntime.Kernels.end()) {
    Kernel = &it->second;
  }
  else {
    auto res = CudaRuntime.Kernels.emplace(
        name,
        KernelData{} );
    Kernel = &res.first->second;
    CudaRuntime.link(Kernel->Module);
    auto err = cuModuleGetFunction(&Kernel->Function, Kernel->Module, name);
    check(err);
  }
  

  auto size = is->size();

  auto err = cuLaunchKernel(
      Kernel->Function,
      1, 1, 1,
      size, 1, 1,
      0,
      nullptr,
      params,
      nullptr);
  check(err);
 
  cudaError_t cudaerr = cudaDeviceSynchronize();
  check(cudaerr, "Kernel launch");

  //err = cuModuleUnload(CuModule);
  //check(err);
}


//==============================================================================
// Copy partition to device
//==============================================================================
void* contra_cuda_2dev(void * ptr, size_t size)
{
  void* dev = nullptr;
  auto err = cudaMalloc(&dev, size);
  check(err, "cudaMalloc");

  err = cudaMemcpy(dev, ptr, size, cudaMemcpyHostToDevice);
  check(err, "cudaMemcpy(Host2Dev)");

  return dev;
}

//==============================================================================
// Copy to host
//==============================================================================
void contra_cuda_2host(void * dev, void * host, size_t size)
{
  auto err = cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
  check(err, "cudaMemcpy(Dev2Host)");
}

//==============================================================================
// Copy array to device
//==============================================================================
dopevector_t contra_cuda_array2dev(dopevector_t * arr)
{
  auto size = arr->bytes();
  auto dev_data = contra_cuda_2dev(arr->data, size);

  dopevector_t dev_arr;
  dev_arr.setup(arr->size, arr->data_size, dev_data);
  return dev_arr;
}

//==============================================================================
// free an array
//==============================================================================
void contra_cuda_array_free(dopevector_t * arr)
{
  auto err = cudaFree(arr->data);
  check(err, "cudaFree");
}


//==============================================================================
// free an array
//==============================================================================
void contra_cuda_free(void ** ptr)
{
  auto err = cudaFree(*ptr);
  check(err, "cudaFree");
}

//==============================================================================
// free an array
//==============================================================================
void contra_cuda_partition_free(contra_cuda_partition_t * part)
{
  auto err = cudaFree(part->offsets);
  check(err, "cudaFree");

  if (part->indices) {
    auto err = cudaFree(part->indices);
    check(err, "cudaFree");
  }

}

//==============================================================================
// Destroy a partition
//==============================================================================
void contra_cuda_partition_destroy(contra_cuda_partition_t * part)
{ part->destroy(); }


//==============================================================================
/// create partition info
//==============================================================================
void contra_cuda_partition_info_create(contra_cuda_partition_info_t** info)
{ *info = new contra_cuda_partition_info_t; }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_cuda_partition_info_destroy(contra_cuda_partition_info_t** info)
{ delete (*info); }


//==============================================================================
// Create a field
//==============================================================================
void contra_cuda_field_create(
    const char * name,
    int_t data_size,
    const void* init,
    contra_index_space_t * is,
    contra_cuda_field_t * fld)
{
  fld->setup(is, data_size);
  auto ptr = static_cast<byte_t*>(fld->data);
  for (int_t i=0; i<is->size(); ++i)
    memcpy(ptr + i*data_size, init, data_size);
}


//==============================================================================
// Destroy a field
//==============================================================================
void contra_cuda_field_destroy(contra_cuda_field_t * fld)
{ fld->destroy(); }

//==============================================================================
/// index space partitioning
//==============================================================================
void contra_cuda_index_space_create_from_partition(
    int_t i,
    contra_cuda_partition_t * part,
    contra_index_space_t * is)
{
  is->setup(part->offsets[i], part->offsets[i+1]);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_cuda_partition_from_size(
    int_t num_parts,
    contra_index_space_t * is,
    contra_cuda_partition_t * part)
{
  auto index_size = is->size();
  auto chunk_size = index_size / num_parts; 
  auto remainder = index_size % num_parts;

  auto offsets = new int_t[num_parts+1];
  offsets[0] = 0;

  for (int_t i=0; i<num_parts; ++i) {
    offsets[i+1] = offsets[i] + chunk_size;
    if (remainder>0) {
      offsets[i+1]++;
      remainder--;
    }
  }

  part->setup( index_size, num_parts, offsets);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_cuda_partition_from_index_space(
    contra_index_space_t * cs,
    contra_index_space_t * is,
    contra_cuda_partition_t * part)
{
  auto num_parts = cs->size();
  contra_cuda_partition_from_size(
      num_parts,
      is,
      part);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_cuda_partition_from_array(
    dopevector_t *arr,
    contra_index_space_t * is,
    contra_cuda_partition_t * part)
{
  auto size_ptr = static_cast<const int_t*>(arr->data);
  int_t expanded_size = 0;
  int_t num_parts = arr->size;
  for (int_t i=0; i<num_parts; ++i) expanded_size += size_ptr[i];

  //------------------------------------
  // if the sizes are differeint 
  auto index_size = is->size();
  if (expanded_size != index_size) {
    std::cerr << "Index spaces partitioned by arrays MUST match the size of "
      << "the original index space." << std::endl;
    abort();  
  }
  //------------------------------------
  // Naive partitioning
  else {
 
    auto offsets = new int_t[num_parts+1];
    offsets[0] = 0;

    for (int_t i=0; i<num_parts; ++i) 
      offsets[i+1] = offsets[i] + size_ptr[i];

    part->setup( index_size, num_parts, offsets);

  }
  //------------------------------------
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_cuda_partition_from_field(
    contra_cuda_field_t *fld,
    contra_index_space_t * is,
    contra_cuda_partition_t * fld_part,
    contra_cuda_partition_t * part)
{
  auto num_parts = fld_part->num_parts;
  auto expanded_size = fld->size;

  auto indices = new int_t[expanded_size];
  auto offsets = new int_t[num_parts+1];
  offsets[0] = 0;
  
  auto fld_ptr = static_cast<int_t*>(fld->data);

  if (auto fld_indices = fld_part->indices) {

    for (int_t i=0, cnt=0; i<num_parts; ++i) {
      auto size = fld_part->part_size(i);
      offsets[i+1] = offsets[i] + size;
      for (int_t j=0; j<size; ++j, ++cnt) {
        indices[cnt] = fld_ptr[ fld_indices[cnt] ]; 
      }
    }

  }
  else {
    
    auto fld_offsets = fld_part->offsets;

    for (int_t i=0, cnt=0; i<num_parts; ++i) {
      auto size = fld_part->part_size(i);
      offsets[i+1] = offsets[i] + size;
      auto fld_part_offset = fld_offsets[i];
      for (int_t j=0; j<size; ++j, ++cnt) {
        indices[cnt] = fld_ptr[ fld_part_offset + j ]; 
      }
    }

  }

  part->setup(expanded_size, num_parts, indices, offsets);
}

//==============================================================================
// Copy partition to device
//==============================================================================
contra_cuda_partition_t contra_cuda_partition2dev(
    contra_index_space_t * host_is,
    contra_cuda_partition_t * part,
    contra_cuda_partition_info_t ** info)
{
  auto num_parts = part->num_parts;
  auto int_size = sizeof(int_t);
  auto offsets = contra_cuda_2dev(part->offsets, (num_parts+1)*int_size);
  
  auto part_size = part->size;
  void* indices = nullptr;
  if (part->indices)
    indices = contra_cuda_2dev(part->indices, part_size*int_size);

  (*info)->register_partition(host_is, part);

  contra_cuda_partition_t dev_part;
  dev_part.setup(
      part_size,
      num_parts,
      static_cast<int_t*>(indices),
      static_cast<int_t*>(offsets));

  return dev_part;
}

//==============================================================================
/// Accessor write
//==============================================================================
contra_cuda_accessor_t contra_cuda_field2dev(
    contra_index_space_t * cs,
    contra_cuda_partition_t * part,
    contra_cuda_field_t * fld,
    contra_cuda_partition_info_t **info)
{
  //------------------------------------
  // Create a partition
  if (!part) {

    // no partitioning specified
    auto res = (*info)->getOrCreatePartition(fld->index_space);
    part = res.first;
    if (!res.second) {
      contra_cuda_partition_from_index_space(
          cs,
          fld->index_space,
          part);
      (*info)->register_partition(fld->index_space, part);
    }

  } // specified part
  
  //------------------------------------
  // copy field over
  void* dev_data = nullptr;
  auto data_size = fld->data_size;
  
  if (part->indices) {
    auto size = part->size;
    auto bytes = data_size * size;
    auto src = static_cast<byte_t*>(fld->data);
    auto tmp = new byte_t[bytes];
    auto dest = tmp;
    for (int_t i=0; i<size; ++i) {
      memcpy(dest, src, data_size);
      dest += data_size;
      src += data_size;
    }
    dev_data = contra_cuda_2dev(tmp, bytes);
    delete [] tmp;
  }
  else {
    dev_data = contra_cuda_2dev(fld->data, fld->bytes());
  }

  auto num_parts = part->num_parts;
  auto int_size = sizeof(int_t);
  auto dev_off = (int_t*)contra_cuda_2dev(part->offsets, (num_parts+1)*int_size);
  
  contra_cuda_field_t * fld_ptr = part->indices ? nullptr : fld;

  contra_cuda_accessor_t dev_acc;
  dev_acc.setup(data_size, dev_data, dev_off, fld_ptr);
  return dev_acc;
}

//==============================================================================
// free an array
//==============================================================================
void contra_cuda_accessor_free(contra_cuda_accessor_t * acc)
{

  // data needs to come back
  if (acc->field) {
    auto fld = acc->field;
    contra_cuda_2host(acc->data, fld->data, fld->bytes());
  }

  auto err = cudaFree(acc->data);
  check(err, "cudaFree");
  err = cudaFree(acc->offsets);
  check(err, "cudaFree");
}

//==============================================================================
// prepare a reduction
//==============================================================================
void contra_cuda_prepare_reduction(
  void ** indata,
  size_t data_size,
  contra_index_space_t * is)
{
  auto size = is->size();
  auto bytes = data_size * size;
  auto err = cudaMalloc(indata, bytes);
  check(err, "cudaMalloc");
}

//==============================================================================
// launch a kernel
//==============================================================================
void contra_cuda_launch_reduction(
    char * kernel_name,
    char * init_name,
    contra_index_space_t * is,
    void ** dev_indata,
    void * outdata)
{

  ReductionlData* Reduction;
  KernelData* Kernel;

  init_t hptr;
  
  //------------------------------------
  // Reductions already setup
  auto it = CudaRuntime.Reductions.find(kernel_name);
  if (it != CudaRuntime.Reductions.end()) {
    Reduction = &it->second;
  }
  //------------------------------------
  // Need to sort them out first
  else {

    auto it = CudaRuntime.Kernels.find(kernel_name);
    if (it != CudaRuntime.Kernels.end()) {
      Kernel = &it->second;
    }
    else {
      std::cerr << "Could not find kernel '" << kernel_name << "'. Reduction "
        << "functions should be packed with it." << std::endl; 
      abort();
    }

    auto res = CudaRuntime.Reductions.emplace(
        kernel_name,
        ReductionlData{} );
    Reduction = &res.first->second;

    auto err = cuModuleGetFunction(&Reduction->InitFunction, Kernel->Module, init_name);
    check(err);


    CUdeviceptr dptr;
    size_t bytes;
    auto cuerr = cuModuleGetGlobal(
      &dptr,
      &bytes,
      Kernel->Module,
      "test");
    check(cuerr);

    auto er = cudaMemcpy((void*)(&hptr), (const void *)dptr, bytes, cudaMemcpyDeviceToHost);
    check(er, "cudaMemcpyDeviceToHost");
  }

  //void * dev_init;
  //auto err = cudaGetSymbolAddress(&dev_init, "test");
  //check(err, "cudaSymbolAddress");

#if 0
  auto err = cudaMalloc(&dev_init, sizeof(init_t));
  std::cout << "device ptr " << dev_init << std::endl;

  void* params[] = {&dev_init};

  auto cuerr = cuLaunchKernel(
      Reduction->InitFunction,
      1, 1, 1,
      1, 1, 1,
      0,
      nullptr,
      params,
      nullptr);
  check(cuerr);
 
  cudaError_t cudaerr = cudaDeviceSynchronize();
  check(cudaerr, "Init reduction launch");
#endif

  size_t size = is->size();
  size_t block_size = 1;
  size_t data_size = sizeof(int_t) + sizeof(real_t);

  void * dev_outdata;
  cudaMalloc(&dev_outdata, data_size); // num blocks
  size_t bytes = data_size * size;
  
  CUfunction ReduceFunction;
  auto err = cuModuleGetFunction(&ReduceFunction, Kernel->Module, "reduce6");
  check(err);
  std::cout << "HERHREHREHER" << std::endl;

  void * params[] = {
    dev_indata,
    &dev_outdata,
    &data_size,
    &size,
    &block_size,
    &hptr
  };

  err = cuLaunchKernel(
      ReduceFunction,
      1, 1, 1,
      size, 1, 1,
      bytes,
      nullptr,
      params,
      nullptr);
  check(err);
  
  auto cudaerr = cudaDeviceSynchronize();
  check(cudaerr, "Reduction launch");

  cudaMemcpy(outdata, dev_outdata, data_size, cudaMemcpyDeviceToHost); 
  cudaFree(dev_outdata);

  //err = cudaFree(dev_init);
  //check(err, "cudaFree");

}



} // extern
