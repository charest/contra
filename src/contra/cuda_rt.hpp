#ifndef CONTRA_CUDA_RT_HPP
#define CONTRA_CUDA_RT_HPP

#include "config.hpp"
#include "tasking_rt.hpp"

#include <cuda.h>

#include <iostream>
#include <map>
#include <vector>

struct KernelData {
  CUfunction Function;
  CUmodule Module;
};

////////////////////////////////////////////////////////////////////////////////
// Runtime definition
////////////////////////////////////////////////////////////////////////////////
struct cuda_runtime_t {
  CUdevice CuDevice;  
  CUcontext CuContext;
  bool IsStarted = false;
 
  std::vector<std::string> Ptxs;

  std::map<std::string, KernelData> Kernels;

  size_t MaxThreadsPerBlock = 0;

  void init(int);
  void shutdown();

  void link(CUmodule &);
  void link_start();
  
  void threadDims(size_t NumThreads, size_t &, size_t &);

};


////////////////////////////////////////////////////////////////////////////////
/// Types needed for cuda runtime
////////////////////////////////////////////////////////////////////////////////
extern "C" {

//==============================================================================
struct contra_cuda_partition_t {
  int_t size;
  int_t num_parts;
  int_t* offsets;
  int_t* indices;

  void setup(
      int_t part_sz,
      int_t parts,
      int_t * offs)
  {
    num_parts = parts;
    size = part_sz;
    offsets = offs;
    indices = nullptr;
  }

  void setup(
      int_t part_sz,
      int_t parts,
      int_t * indx,
      int_t * offs)
  {
    num_parts = parts;
    size = part_sz;
    offsets = offs;
    indices = indx;
  }

  void destroy() {
    if (offsets) {
      delete[] offsets;
      offsets = nullptr;
    }
    if (indices) {
      delete[] indices;
      indices = nullptr;
    }
    size = 0;
    num_parts = 0;
  }

  CONTRA_INLINE_TARGET int_t part_size(int_t i) { return offsets[i+1] - offsets[i]; }
  
};


//==============================================================================
struct contra_cuda_field_t {
  int_t data_size;
  int_t size;
  void *data;
  contra_index_space_t * index_space;

  void setup(
      contra_index_space_t *is,
      int_t data_sz)
  {
    size = is->size();
    data_size = data_sz;
    data = malloc(data_size*size);
    index_space = is;
  }

  void destroy() {
    free(data);
    data_size = 0;
    size = 0;
    data = nullptr;
    index_space = nullptr;
  }

  int_t bytes() { return data_size*size; }
};

//==============================================================================
struct contra_cuda_accessor_t {
  int_t data_size;
  void *data;
  contra_cuda_partition_t partition;
  contra_cuda_field_t * field;
  
  void setup(
    int_t data_sz,
    void *dat,
    contra_cuda_partition_t pt,
    contra_cuda_field_t * f)
  {
    data_size = data_sz;
    data = dat;
    partition = pt;
    field = f;
  }
};


//==============================================================================
struct contra_cuda_partition_info_t {
  std::map<contra_index_space_t*, contra_cuda_partition_t*> IndexPartMap;
  std::vector<contra_cuda_partition_t*> PartsToDelete;

  void register_partition(
      contra_index_space_t * is,
      contra_cuda_partition_t * part)
  { IndexPartMap.emplace(is, part); }

  std::pair<contra_cuda_partition_t*, bool>
    getOrCreatePartition(contra_index_space_t * is)
  {
    auto it = IndexPartMap.find(is);
    if (it != IndexPartMap.end()) {
      return {it->second, true};
    }
    else {
      auto part = new contra_cuda_partition_t;
      PartsToDelete.push_back(part);
      return {part, false};
    }
  }
  
  ~contra_cuda_partition_info_t() {
    for (auto part : PartsToDelete)
      part->destroy();
  }
};

//==============================================================================
struct contra_cuda_task_info_t {

  std::map<contra_cuda_partition_t*, contra_cuda_partition_t> Host2DevPart;
  std::map<contra_cuda_partition_t*, contra_cuda_partition_t*> Dev2HostPart;
  
  std::map<void*, void*> Host2DevField;
  
  std::map<
    std::pair<contra_cuda_partition_t*, contra_cuda_field_t*>,
    contra_cuda_accessor_t> Host2DevAcc;
  std::map<
    contra_cuda_accessor_t*,
    std::pair<contra_cuda_partition_t*, contra_cuda_field_t*> > TempDev2HostAcc;

  std::pair<contra_cuda_partition_t*, bool> 
    getOrCreatePartition(contra_cuda_partition_t*);
  
  std::pair<contra_cuda_accessor_t*, bool> 
    getOrCreateAccessor(contra_cuda_partition_t*, contra_cuda_field_t*, bool);
  
  std::pair<contra_cuda_accessor_t*, bool> 
    getAccessor(contra_cuda_partition_t*, contra_cuda_field_t*);
  
  std::pair<void**, bool> 
    getOrCreateField(void*);


  void freePartition(contra_cuda_partition_t*);
  void freeTempAccessor(contra_cuda_accessor_t*);

  bool isOnDevice(contra_cuda_partition_t* part)
  { return Host2DevPart.count(part); }

  bool isOnDevice(void* data)
  { return Host2DevField.count(data); }

  ~contra_cuda_task_info_t();
};

////////////////////////////////////////////////////////////////////////////////
// public functions
////////////////////////////////////////////////////////////////////////////////

void contra_cuda_startup();
void contra_cuda_shutdown();
void contra_cuda_register_kernel(const char * kernel);

void contra_cuda_partition_free(contra_cuda_partition_t *);
void contra_cuda_accessor_free(contra_cuda_accessor_t *);

} // extern


#endif // LIBRT_LEGION_RT_HPP
