#include "config.hpp"

#include "rocm_rt.hpp"
#include "tasking_rt.hpp"

#include "librt/dopevector.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

//==============================================================================
// Utility check function
//==============================================================================
void check(hipError_t err){
  if(err != hipSuccess){
    auto n = hipGetErrorName(err);
    auto s = hipGetErrorString(err);
    std::cerr << n << " error: " << s << std::endl;
    abort();
  }  
}                                                                                               

#if 0
void check(cudaError_t err, const char * msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " failed with error \""
      << cudaGetErrorString(err) << "\"." << std::endl;
    abort();
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Runtime definition
////////////////////////////////////////////////////////////////////////////////

extern "C" {

/// global runtime for compiled cases
rocm_runtime_t RocmRuntime;

//==============================================================================
// Get/create memory for an index partition
//==============================================================================
std::pair<contra_rocm_partition_t*, bool>
  contra_rocm_task_info_t::getOrCreatePartition(contra_rocm_partition_t* host)
{
  // found
  auto it = Host2DevPart.find(host);
  if (it!=Host2DevPart.end()) {
    return std::make_pair(&it->second, true);
  }
  // not found
  auto res = Host2DevPart.emplace(host, contra_rocm_partition_t{});
  auto dev_ptr = &res.first->second;
  Dev2HostPart.emplace(dev_ptr, host);
  return std::make_pair(dev_ptr, false);
}

//==============================================================================
// Free a partition
//==============================================================================
void contra_rocm_task_info_t::freePartition(contra_rocm_partition_t* dev)
{
  auto it = Dev2HostPart.find(dev);
  if (it != Dev2HostPart.end()) {
    contra_rocm_partition_free(dev);
    auto host = it->second;
    Dev2HostPart.erase(it);
    Host2DevPart.erase(host);
  }
}
  
//==============================================================================
// create an accessor
//==============================================================================
std::pair<contra_rocm_accessor_t*, bool> 
contra_rocm_task_info_t::getOrCreateAccessor(
    contra_rocm_partition_t* part,
    contra_rocm_field_t* field,
    bool IsTemporary)
{
  // found
  auto it = Host2DevAcc.find({part, field});
  if (it!=Host2DevAcc.end()) {
    return std::make_pair(&it->second, true);
  }
  // not found
  auto res = Host2DevAcc.emplace(
      std::make_pair(part, field),
      contra_rocm_accessor_t{});
  auto dev_ptr = &res.first->second;
  if (IsTemporary) TempDev2HostAcc.emplace(dev_ptr, std::make_pair(part,field));
  return std::make_pair(dev_ptr, false);
}

//==============================================================================
// get an accessor
//==============================================================================
std::pair<contra_rocm_accessor_t*, bool> 
contra_rocm_task_info_t::getAccessor(
    contra_rocm_partition_t* part,
    contra_rocm_field_t* field)
{
  // found
  auto it = Host2DevAcc.find({part, field});
  if (it!=Host2DevAcc.end()) {
    return std::make_pair(&it->second, true);
  }
  return std::make_pair(nullptr, false);
  // not found
}

//==============================================================================
// Free a temporary accessor
//==============================================================================
void contra_rocm_task_info_t::freeTempAccessor(contra_rocm_accessor_t* dev)
{
  auto it = TempDev2HostAcc.find(dev);
  if (it != TempDev2HostAcc.end()) {
    contra_rocm_accessor_free(dev);
    auto host = it->second;
    TempDev2HostAcc.erase(it);
    Host2DevAcc.erase(host);
    contra_rocm_accessor_free(dev);
  }
}
  
//==============================================================================
// create field data
//==============================================================================
std::pair<void**, bool> 
contra_rocm_task_info_t::getOrCreateField(void* host)
{
  // found
  auto it = Host2DevField.find(host);
  if (it!=Host2DevField.end()) {
    return std::make_pair(&it->second, true);
  }
  // not found
  auto res = Host2DevField.emplace(host, nullptr);
  auto dev_ptr = &res.first->second;
  return std::make_pair(dev_ptr, false);
}


//==============================================================================
// Destructor for task info
//==============================================================================
contra_rocm_task_info_t::~contra_rocm_task_info_t() {
  for (auto & DevPart : Host2DevPart)
    contra_rocm_partition_free(&DevPart.second);
}
  

//==============================================================================
// start runtime
//==============================================================================
void rocm_runtime_t::init(int dev_id) {
  auto err = hipSetDevice(dev_id);
  check(err);
  printf( "Selected Device: %d\n", dev_id);
    
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, dev_id);
  MaxThreadsPerBlock = props.maxThreadsPerBlock;

  //err = cuCtxCreate(&CuContext, 0, CuDevice);
  //check(err);
  
  IsStarted = true;
}

//==============================================================================
// shutdown runtime
//==============================================================================
void rocm_runtime_t::shutdown() {
  if (!IsStarted) return;

#if 0
  auto err = cuCtxDestroy(CuContext);
  check(err);
#endif

  IsStarted = false;
}
  
  
////////////////////////////////////////////////////////////////////////////////
/// Public c interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// start runtime
//==============================================================================
void contra_rocm_startup() {
  const int kb = 1024;
  const int mb = kb * kb;

  printf( "HIP version:   v%d\n", HIP_VERSION );

  int devCount;
  hipGetDeviceCount(&devCount);
  printf( "HIP Devices: \n\n" );

  for(int i = 0; i < devCount; ++i)
  {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, i);
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

  RocmRuntime.init(0);

  fflush(stdout);
}

//==============================================================================
// start runtime
//==============================================================================
void contra_rocm_shutdown() {
  RocmRuntime.shutdown();
}

//==============================================================================
// jit and register a kernel
//==============================================================================
void contra_rocm_register_kernel(
    const char * kernel,
    size_t size_kernel,
    const char * names[],
    unsigned size_names)
{
  RocmRuntime.Kernels.emplace_back( kernel, kernel+size_kernel );
  auto KernelId = RocmRuntime.Kernels.size();

  for (unsigned i=0; i<size_names; ++i)
    RocmRuntime.KernelMap[names[i]] = i;
}


//==============================================================================
// launch a kernel
//==============================================================================
void contra_rocm_launch_kernel(
    char * name,
    contra_index_space_t * is)
{
  std::cout << "launching " << name << std::endl;
  auto KernelIt = RocmRuntime.KernelMap.find(name);
  if (KernelIt == RocmRuntime.KernelMap.end()) {
    std::cerr << "Did not find a compiled kernel for '" << name << "'";
    abort();
  }

  auto KernelData = RocmRuntime.Kernels[KernelIt->second];
  
  hipModule_t Module;
  auto err = hipModuleLoadData(&Module, (const void*)KernelData.data());
  check(err);

  hipFunction_t F;
  err = hipModuleGetFunction(&F, Module, name);
  check(err);

  auto size = is->size();
  auto Dims = RocmRuntime.threadDims(size);

  
  int_t arg = 1;
  size_t arg_size = sizeof(int_t);


  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &arg,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
    HIP_LAUNCH_PARAM_END
  };

  hipModuleLaunchKernel(
      F,
      Dims.first, 1, 1,
      Dims.second, 1, 1,
      0,
      0,
      nullptr,
      nullptr);
      //config);
  hipDeviceSynchronize();

  err = hipGetLastError();
  check(err);

  err = hipModuleUnload(Module);
  check(err);

  std::cout << "done " << name << std::endl;
 
}


//==============================================================================
// Copy partition to device
//==============================================================================
void* contra_rocm_2dev(void * ptr, size_t size)
{
#if 0
  void* dev = nullptr;
  auto err = cudaMalloc(&dev, size);
  check(err, "cudaMalloc");

  err = cudaMemcpy(dev, ptr, size, cudaMemcpyHostToDevice);
  check(err, "cudaMemcpy(Host2Dev)");

  return dev;
#endif
}

//==============================================================================
// Copy to host
//==============================================================================
void contra_rocm_2host(void * dev, void * host, size_t size)
{
  //auto err = cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
  //check(err, "cudaMemcpy(Dev2Host)");
}

//==============================================================================
// Copy array to device
//==============================================================================
dopevector_t contra_rocm_array2dev(dopevector_t * arr)
{
  auto size = arr->bytes();
  auto dev_data = contra_rocm_2dev(arr->data, size);

  dopevector_t dev_arr;
  dev_arr.setup(arr->size, arr->data_size, dev_data);
  return dev_arr;
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_free(void ** ptr)
{
  //auto err = cudaFree(*ptr);
  //check(err, "cudaFree");
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_array_free(dopevector_t * arr)
{
  //auto err = cudaFree(arr->data);
  //check(err, "cudaFree");
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_partition_free(contra_rocm_partition_t * part)
{
#if 0
  auto err = cudaFree(part->offsets);
  check(err, "cudaFree");

  if (part->indices) {
    auto err = cudaFree(part->indices);
    check(err, "cudaFree");
  }
#endif
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_partition_free_and_deregister(
    contra_rocm_partition_t * part,
    contra_rocm_task_info_t ** task_info)
{
  (*task_info)->freePartition(part);
}

//==============================================================================
// Destroy a partition
//==============================================================================
void contra_rocm_partition_destroy(contra_rocm_partition_t * part)
{ part->destroy(); }


//==============================================================================
/// create partition info
//==============================================================================
void contra_rocm_partition_info_create(contra_rocm_partition_info_t** info)
{ *info = new contra_rocm_partition_info_t; }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_rocm_partition_info_destroy(contra_rocm_partition_info_t** info)
{ delete (*info); }

//==============================================================================
/// create partition info
//==============================================================================
void contra_rocm_task_info_create(contra_rocm_task_info_t** info)
{ *info = new contra_rocm_task_info_t; }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_rocm_task_info_destroy(contra_rocm_task_info_t** info)
{ delete (*info); }


//==============================================================================
// Create a field
//==============================================================================
void contra_rocm_field_create(
    const char * name,
    int_t data_size,
    const void* init,
    contra_index_space_t * is,
    contra_rocm_field_t * fld)
{
  fld->setup(is, data_size);
  auto ptr = static_cast<byte_t*>(fld->data);
  for (int_t i=0; i<is->size(); ++i)
    memcpy(ptr + i*data_size, init, data_size);
}


//==============================================================================
// Destroy a field
//==============================================================================
void contra_rocm_field_destroy(contra_rocm_field_t * fld)
{ fld->destroy(); }

//==============================================================================
/// index space partitioning
//==============================================================================
void contra_rocm_index_space_create_from_partition(
    int_t i,
    contra_rocm_partition_t * part,
    contra_index_space_t * is)
{
  is->setup(part->offsets[i], part->offsets[i+1]);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_rocm_partition_from_size(
    int_t num_parts,
    contra_index_space_t * is,
    contra_rocm_partition_t * part)
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
void contra_rocm_partition_from_index_space(
    contra_index_space_t * cs,
    contra_index_space_t * is,
    contra_rocm_partition_t * part)
{
  auto num_parts = cs->size();
  contra_rocm_partition_from_size(
      num_parts,
      is,
      part);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_rocm_partition_from_array(
    dopevector_t *arr,
    contra_index_space_t * is,
    contra_rocm_partition_t * part)
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
void contra_rocm_partition_from_field(
    contra_rocm_field_t *fld,
    contra_index_space_t * is,
    contra_rocm_partition_t * fld_part,
    contra_rocm_partition_t * part,
    contra_rocm_task_info_t ** task_info)
{
  auto num_parts = fld_part->num_parts;
  auto expanded_size = fld->size;

  auto indices = new int_t[expanded_size];
  auto offsets = new int_t[num_parts+1];
  offsets[0] = 0;
    
  auto acc_res = (*task_info)->getAccessor(fld_part, fld);
  if (acc_res.second) {
    auto acc = acc_res.first;
    contra_rocm_2host(acc->data, fld->data, fld->bytes());
  }
  
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
contra_rocm_partition_t contra_rocm_partition2dev(
    contra_index_space_t * host_is,
    contra_rocm_partition_t * part,
    contra_rocm_partition_info_t ** part_info,
    contra_rocm_task_info_t ** task_info)
{
  // link index space to partition
  (*part_info)->register_partition(host_is, part);

  // is this already on the device
  auto res = (*task_info)->getOrCreatePartition(part);
  auto dev_part = res.first;

  // not on device, create
  if (!res.second) {
  
    auto num_parts = part->num_parts;
    auto int_size = sizeof(int_t);
    auto offsets = contra_rocm_2dev(part->offsets, (num_parts+1)*int_size);
  
    auto part_size = part->size;
    void* indices = nullptr;
    if (part->indices)
    indices = contra_rocm_2dev(part->indices, part_size*int_size);

    dev_part->setup(
        part_size,
        num_parts,
        static_cast<int_t*>(indices),
        static_cast<int_t*>(offsets));
  
  }

  return *dev_part;
}
        
//==============================================================================
/// create an accessor
//==============================================================================
void contra_rocm_accessor_2dev(
    contra_rocm_partition_t * part,
    contra_rocm_field_t * fld,
    contra_rocm_partition_t dev_part,
    void * dev_data,
    contra_rocm_accessor_t *acc)
{
  auto data_size = fld->data_size;
  contra_rocm_field_t * fld_ptr = part->indices ? nullptr : fld;
  acc->setup(data_size, dev_data, dev_part, fld_ptr);
}

//==============================================================================
/// Accessor write
//==============================================================================
contra_rocm_accessor_t contra_rocm_field2dev(
    contra_index_space_t * cs,
    contra_rocm_partition_t * part,
    contra_rocm_field_t * fld,
    contra_rocm_partition_info_t **part_info,
    contra_rocm_task_info_t **task_info)
{
  bool is_temporary = part;
#if 0

  //----------------------------------------------------------------------------
  // Create a partition
  if (!part) {

    // no partitioning specified
    auto res = (*part_info)->getOrCreatePartition(fld->index_space);
    part = res.first;
    if (!res.second) {
      contra_rocm_partition_from_index_space(
          cs,
          fld->index_space,
          part);
      (*part_info)->register_partition(fld->index_space, part);
    }

  } // specified part

  // move it onto the device
  auto dev_part = contra_rocm_partition2dev(
      fld->index_space,
      part,
      part_info,
      task_info);
  
  //----------------------------------------------------------------------------
  // copy field over
    
  // get field
  auto fld_res = (*task_info)->getOrCreateField(fld->data);
  auto dev_data = fld_res.first;
  if (!fld_res.second)
    *dev_data = contra_rocm_2dev(fld->data, fld->bytes());
  
  // look for a partition/field combo
  auto acc_res = (*task_info)->getOrCreateAccessor(part, fld, is_temporary);
  auto dev_acc = acc_res.first;
  
  //------------------------------------
  // overlapping part
  if (auto indices = part->indices) {
    auto data_size = fld->data_size;
    auto size = part->size;
    auto bytes = data_size * size;
    if (!acc_res.second) {
      //auto err = cudaMalloc(&(dev_acc->data), bytes);
      //check(err, "cudaMalloc");
    }
    auto Dims = CudaRuntime.threadDims(size);
    rocm_copy(
      static_cast<byte_t*>(*dev_data),
      static_cast<byte_t*>(dev_acc->data),
      dev_part.indices,
      data_size,
      size,
      Dims.first,
      Dims.second);
    
    // move accessor over
    if (!acc_res.second)
        contra_rocm_accessor_2dev(part, fld, dev_part, dev_acc->data, dev_acc);
  }
  //------------------------------------
  // Disjoint partitiong
  else {
    // move accessor over
    if (!acc_res.second)
        contra_rocm_accessor_2dev(part, fld, dev_part, *dev_data, dev_acc);
  }


  return *dev_acc;
#endif
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_accessor_free(contra_rocm_accessor_t * acc)
{

  // data needs to come back
  if (acc->field) {
    auto fld = acc->field;
    contra_rocm_2host(acc->data, fld->data, fld->bytes());
  }

  //auto err = cudaFree(acc->data);
  //check(err, "cudaFree");
  //auto err = cudaFree(acc->offsets);
  //check(err, "cudaFree");
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_accessor_free_temp(
    contra_rocm_accessor_t * acc,
    contra_rocm_task_info_t ** task_info)
{
  (*task_info)->freeTempAccessor(acc);
}
  
  
//==============================================================================
// prepare a reduction
//==============================================================================
void contra_rocm_prepare_reduction(
  void ** indata,
  size_t data_size,
  contra_index_space_t * is)
{
  auto size = is->size();
  auto bytes = data_size * size;
  //auto err = cudaMalloc(indata, bytes);
  //check(err, "cudaMalloc");
}

//==============================================================================
// Get a global symbol
//==============================================================================
#if 0  
void contra_rocm_get_symbol(
    const char * name,
    CUmodule * CuModule,
    void * ptr,
    size_t size)
{
  CUdeviceptr dev_ptr;
  auto err = cuModuleGetGlobal(
    &dev_ptr,
    0,
    *CuModule,
    name);
  check(err);

  auto cuerr = cudaMemcpy(
      ptr,
      (const void *)dev_ptr,
      size,
      cudaMemcpyDeviceToHost);
  check(cuerr, "cudaMemcpyDeviceToHost");
}
#endif

//==============================================================================
// launch a kernel
//==============================================================================
void contra_rocm_launch_reduction(
    char * kernel_name,
    char * init_name,
    char * apply_name,
    char * fold_name,
    contra_index_space_t * is,
    void ** dev_indata,
    void * result,
    size_t data_size)//,
    //apply_t host_apply)
{
#if 0
  KernelData* Kernel;
  auto it = CudaRuntime.Kernels.find(kernel_name);
  if (it != CudaRuntime.Kernels.end()) {
    Kernel = &it->second;
  }
  else {
    std::cerr << "Could not find kernel '" << kernel_name << "'. Reduction "
      << "functions should be packed with it." << std::endl; 
    abort();
  }

  // get device pointers
  init_t init_ptr;
  contra_rocm_get_symbol(
    init_name,
    &Kernel->Module,
    &init_ptr,
    sizeof(init_t));

  apply_t apply_ptr;
  contra_rocm_get_symbol(
    apply_name,
    &Kernel->Module,
    &apply_ptr,
    sizeof(apply_t));

  fold_t fold_ptr;
  contra_rocm_get_symbol(
    fold_name,
    &Kernel->Module,
    &fold_ptr,
    sizeof(fold_t));

  // some dimensinos
  size_t size = is->size();
  size_t bytes = data_size * size;
  
  // block dimensinos
  auto Dims = CudaRuntime.threadDims(size);
  size_t block_size = Dims.first;
  size_t threads_per_block = Dims.second;

  // block level storage for result
  void * dev_outdata;
  cudaMalloc(&dev_outdata, block_size*data_size); // num blocks
  
  // Get the reduction function
  CUfunction ReduceFunction;
  auto err = cuModuleGetFunction(&ReduceFunction, Kernel->Module, "reduce6");
  check(err);

  // launch the final reduction
  void * params[] = {
    dev_indata,
    &dev_outdata,
    &data_size,
    &size,
    &block_size,
    &init_ptr,
    &apply_ptr,
    &fold_ptr
  };

  err = cuLaunchKernel(
      ReduceFunction,
      block_size, 1, 1,
      threads_per_block, 1, 1,
      bytes,
      nullptr,
      params,
      nullptr);
  check(err);
  
  auto cuerr = cudaDeviceSynchronize();
  check(cuerr, "Reduction launch");

  // copy over data
  if (block_size > 1) {
    auto outdata = malloc(block_size*data_size);
    auto err = cudaMemcpy(outdata, dev_outdata, block_size*data_size, cudaMemcpyDeviceToHost);
    check(err, "cudaMemcpy");
    auto outdata_bytes = static_cast<byte_t*>(outdata);
    for (unsigned i=1; i<block_size; ++i)
      (host_apply)( outdata_bytes, outdata_bytes+data_size*i );
    memcpy(result, outdata, data_size);
    free(outdata);
  }
  else {
    cudaMemcpy(result, dev_outdata, block_size*data_size, cudaMemcpyDeviceToHost); 
  }
  cudaFree(dev_outdata);

  // final reduce
  
#endif
}



} // extern
