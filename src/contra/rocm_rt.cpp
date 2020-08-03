#include "config.hpp"

#include "rocm_rt.hpp"
#include "tasking_rt.hpp"

#include "librt/dopevector.hpp"
#include "librtrocm/rocm_utils.hpp"

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

////////////////////////////////////////////////////////////////////////////////
/// Runtime definition
////////////////////////////////////////////////////////////////////////////////

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
  //MaxThreadsPerBlock = 512;
  
  IsStarted = true;
}

//==============================================================================
// shutdown runtime
//==============================================================================
void rocm_runtime_t::shutdown() {
  if (!IsStarted) return;

  IsStarted = false;
}
  
//==============================================================================
// load a kernel
//==============================================================================
const rocm_runtime_t::KernelData & rocm_runtime_t::loadKernel(
    const char * name,
    hipModule_t * M,
    hipFunction_t * F)
{
  auto KernelIt = RocmRuntime.KernelMap.find(name);
  if (KernelIt == RocmRuntime.KernelMap.end()) {
    std::cerr << "Did not find a compiled kernel for '" << name << "'";
    abort();
  }

  auto & KernelData = RocmRuntime.Kernels.at(KernelIt->second);

  auto err = hipModuleLoadData(M, (const void*)KernelData.Hsaco.data());
  check(err);
  
  err = hipModuleGetFunction(F, *M, name );
  check(err);

  return KernelData;
}

//==============================================================================
// Determine the number of threads/blocks
//==============================================================================
void rocm_runtime_t::threadDims(
    size_t NumThreads,
    size_t & NumBlocks,
    size_t & ThreadsPerBlock
    )
{
  NumBlocks = 1;
  ThreadsPerBlock = NumThreads;

  if (NumThreads > MaxThreadsPerBlock) {
    ThreadsPerBlock = MaxThreadsPerBlock;
    NumBlocks = NumThreads / ThreadsPerBlock;
    if (NumThreads % ThreadsPerBlock) NumBlocks++;
  }
}

  
////////////////////////////////////////////////////////////////////////////////
/// Public c interface
////////////////////////////////////////////////////////////////////////////////
extern "C" {

void contra_rocm_set_block_size(size_t n) { RocmRuntime.setMaxBlockSize(n); }

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
    unsigned size_names,
    bool HasReduce)
{
  auto KernelId = RocmRuntime.Kernels.size();

  rocm_runtime_t::KernelData Data;
  Data.Hsaco.assign(kernel, kernel+size_kernel);
  Data.HasReduce = HasReduce;
  
  RocmRuntime.Kernels.emplace_back( std::move(Data) );

  for (unsigned i=0; i<size_names; ++i) 
    RocmRuntime.KernelMap[names[i]] = KernelId;
}

//==============================================================================
// launch a kernel
//==============================================================================
void contra_rocm_launch_kernel(
    char * name,
    contra_index_space_t * is,
    void * args,
    size_t args_size,
    void * result,
    size_t result_size,
    void ** dev_indata,
    void ** dev_outdata,
    void(*host_fold)(byte_t*, byte_t*))
{
  hipFunction_t F;
  hipModule_t M;
  RocmRuntime.loadKernel(name, &M, &F);

  auto size = is->size();
  size_t num_blocks, num_threads_per_block;
  RocmRuntime.threadDims(size, num_blocks, num_threads_per_block);
  
  // size of shared memory
  size_t shared_bytes = result_size * num_threads_per_block;

  // allocate thread level storage for reduction
  if (result_size) {
    auto err = hipMalloc(dev_indata, size*result_size);
    check(err);
    err = hipMalloc(dev_outdata, num_blocks*result_size);
    check(err);
  }

  // launch
  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
    HIP_LAUNCH_PARAM_END
  };
  hipModuleLaunchKernel(
      F,
      num_blocks, 1, 1,
      num_threads_per_block, 1, 1,
      shared_bytes,
      0,
      nullptr,
      config);
  //hipDeviceSynchronize();

  auto err = hipGetLastError();
  check(err);

  err = hipModuleUnload(M);
  check(err);

  if (result_size == 0) return;
  
  //----------------------------------------------------------------------------
  // do reduction

  //------------------------------------
  // If block size is 1, its simple
  if (size==1) {
    auto err = hipMemcpy(result, *dev_outdata, result_size, hipMemcpyDeviceToHost); 
    check(err);
    return;
  }

  //------------------------------------
  // Otherwise, apply the reduce
  else {
    if (num_blocks > 1) {
      auto outdata = malloc(num_blocks*result_size);
      auto err = hipMemcpy(outdata, *dev_outdata, num_blocks*result_size, hipMemcpyDeviceToHost);
      check(err);
      auto outdata_bytes = static_cast<byte_t*>(outdata);
      for (unsigned i=1; i<num_blocks; ++i)
        (host_fold)( outdata_bytes, outdata_bytes+result_size*i );
      memcpy(result, outdata, result_size);
      free(outdata);
    }
    else {
      auto err = hipMemcpy(result, *dev_outdata, result_size, hipMemcpyDeviceToHost); 
      check(err);
    }
  }

  // clean up  
  hipFree(*dev_indata);
  hipFree(*dev_outdata);
}


//==============================================================================
// Copy partition to device
//==============================================================================
void* contra_rocm_2dev(void * ptr, size_t size)
{
  void* dev = nullptr;
  auto err = hipMalloc(&dev, size);
  check(err);

  err = hipMemcpy(dev, ptr, size, hipMemcpyHostToDevice);
  check(err);

  return dev;
}

//==============================================================================
// Copy to host
//==============================================================================
void contra_rocm_2host(void * dev, void * host, size_t size)
{
  auto err = hipMemcpy(host, dev, size, hipMemcpyDeviceToHost);
  check(err);
}

//==============================================================================
// Copy array to device
//==============================================================================
void contra_rocm_array2dev(dopevector_t * arr, dopevector_t * dev_arr)
{
  auto size = arr->bytes();
  auto dev_data = contra_rocm_2dev(arr->data, size);

  dev_arr->setup(arr->size, arr->data_size, dev_data);
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_free(void ** ptr)
{
  auto err = hipFree(*ptr);
  check(err);
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_array_free(dopevector_t * arr)
{
  auto err = hipFree(arr->data);
  check(err);
}

//==============================================================================
// free an array
//==============================================================================
void contra_rocm_partition_free(contra_rocm_partition_t * part)
{
  auto err = hipFree(part->offsets);
  check(err);

  if (part->indices) {
    auto err = hipFree(part->indices);
    check(err);
  }
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
      auto err = hipMalloc(&(dev_acc->data), bytes);
      check(err);
    }

    size_t num_blocks, num_threads_per_block;
    RocmRuntime.threadDims(size, num_blocks, num_threads_per_block);
    rocm_copy_kernel<<<num_blocks, num_threads_per_block>>>(
      static_cast<byte_t*>(*dev_data),
      static_cast<byte_t*>(dev_acc->data),
      dev_part.indices,
      data_size,
      size);
    
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

} // extern
