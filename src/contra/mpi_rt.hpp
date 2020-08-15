#ifndef CONTRA_MPI_RT_HPP
#define CONTRA_MPI_RT_HPP

#include "config.hpp"
#include "tasking_rt.hpp"

#include "librt/dopevector.hpp"

#include <mpi.h>

#include <cstring>
#include <iostream>
#include <map>
#include <vector>

namespace contra {

struct field_registry_t {
  std::shared_ptr<void> Init;

  field_registry_t(int_t data_size, const void * init)
  {
    if (init) {
      Init = std::shared_ptr<void>(malloc(data_size), free);
      memcpy(Init.get(), init, data_size);
    }
  }

  auto getInit() const { return Init.get(); }
  
};

struct field_exchange_t {
  void* RecvBuf = nullptr;
  std::vector<MPI_Request> Requests;
  

  void setup(int_t recvsize, int_t reqsize)
  {
    RecvBuf = malloc(recvsize);
    Requests.reserve(reqsize);
  }

  auto getBuffer() const { return RecvBuf; }
  auto & getRequests() { return Requests; }

  auto transferBuffer() {
    auto buf = RecvBuf;
    RecvBuf = nullptr;
    return buf;
  }

  ~field_exchange_t() {
    if (RecvBuf) free(RecvBuf);
    RecvBuf = nullptr;
  }
};

////////////////////////////////////////////////////////////////////////////////
/// mpi runtime
////////////////////////////////////////////////////////////////////////////////
class mpi_runtime_t {

  int Rank = -1;
  int Size = 0;
  unsigned TaskCounter = 0;
  
  std::vector<field_registry_t> FieldRegistry;
  std::map<void*, field_exchange_t> FieldRequests;


public:
  
  void setup(int rank, int size)
  {
    Rank = rank;
    Size = size;
  }

  void incrementCounter() { TaskCounter++; }
  void decrementCounter() { TaskCounter--; }
  auto getCounter() { return TaskCounter; }

  void check(int);
  bool isRoot() const { return Rank == 0; }

  auto getSize() const { return Size; }
  auto getRank() const { return Rank; }

  int registerField(int_t data_size, const void * init)
  {
    auto fid = FieldRegistry.size();
    FieldRegistry.emplace_back(data_size, init);
    return fid;
  }

  auto & getField(int i) { return FieldRegistry.at(i); }

  auto & requestField(void * key, int_t recvcnt, int_t reqcnt) 
  { 
    auto & obj = FieldRequests[key];
    obj.setup(recvcnt, reqcnt);
    return obj;
  }

  std::pair<field_exchange_t*, bool> findRequest(void* data)
  {
    auto it = FieldRequests.find(data);
    if (it != FieldRequests.end()) {
      return {&it->second, true};
    }
    else {
      return {nullptr, false};
    }
  }
};

} // namespace

extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Types needed for mpi runtime
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
struct contra_mpi_partition_t {
  int_t part_size;
  int_t num_parts;
  int_t* offsets;
  int_t* indices;
  contra_index_space_t *index_space;

  void setup(
      int_t part_sz,
      int_t parts,
      contra_index_space_t *is,
      int_t * offs)
  {
    num_parts = parts;
    part_size = part_sz;
    index_space = is;
    offsets = offs;
    indices = nullptr;
  }

  void setup(
      int_t part_sz,
      int_t parts,
      contra_index_space_t *is,
      int_t * indx,
      int_t * offs)
  {
    num_parts = parts;
    part_size = part_sz;
    index_space = is;
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
    part_size = 0;
    num_parts = 0;
    index_space = nullptr;
  }

  auto size(int_t i) { return offsets[i+1] - offsets[i]; }

  auto offsets_begin() { return offsets; }
  auto offsets_end() { return offsets + num_parts; }
  
};


//==============================================================================
struct contra_mpi_field_t {
  int_t data_size = 0;
  void *data = nullptr;
  contra_index_space_t *index_space = nullptr;
  int id = -1;
  int_t * distribution = nullptr;
  int_t * offsets = nullptr;
  int_t num_offsets = 0;
  int_t begin = 0;
  int_t end = 0;

  void setup(
      contra_index_space_t *is,
      int_t data_sz,
      int fid)
  {
    data_size = data_sz;
    data = nullptr;
    index_space = is;
    id = fid;
    distribution = nullptr;
    offsets = nullptr;
    num_offsets = 0;
    begin = 0;
    end = 0;
  }

  void destroy() {
    if (data) free(data);
    data_size = 0;
    data = nullptr;
    index_space = nullptr;
    id = -1;
    if (offsets) delete[] offsets;
    if (distribution) delete[] distribution;
    num_offsets = 0;
    begin = 0;
    end = 0;
  }

  int_t allocate(contra_mpi_partition_t *part, int_t *dist, int_t rank, int_t size)
  {
    auto num_parts = part->num_parts; 
    auto part_offsets = part->offsets;
    
    num_offsets = num_parts;
    offsets = new int_t[num_offsets+1];
    memcpy(offsets, part_offsets, (num_offsets+1)*sizeof(int_t));
    
    distribution = new int_t[size+1];
    memcpy(distribution, dist, (size+1)*sizeof(int_t));
    
    begin = std::min(dist[rank], num_parts);
    end = std::min(dist[rank+1], num_parts);
    auto len = part_offsets[end] - part_offsets[begin];
    data = malloc(data_size*len);
    return len;
  }

  void redistribute(contra_mpi_partition_t *part, int_t *dist, int_t rank, int_t size)
  {
    auto num_parts = part->num_parts;
    if (num_parts > num_offsets) {
      delete[] offsets;
      offsets = new int_t[num_parts];
    }
    num_offsets = num_parts;
    memcpy(offsets, part->offsets, (num_offsets+1)*sizeof(int_t));
    memcpy(distribution, dist, (size+1)*sizeof(int_t));

    begin = std::min(dist[rank], num_parts);
    end = std::min(dist[rank+1], num_parts);
  }

  void transfer(void * buf) {
    if (data) free(data);
    data = buf;
  }
  
  int_t begin_offset() { return offsets[begin]; }
  int_t end_offset() { return offsets[end]; }
  
  int_t begin_offset(int_t i) { return offsets[distribution[i]]; }
  int_t end_offset(int_t i) { return offsets[distribution[i+1]]; }


  bool is_allocated() { return data; }
};

//==============================================================================
struct contra_mpi_accessor_t {
  bool is_allocated;
  int_t data_size;
  void *data;
  
  void setup(void * ptr, int_t data_sz) {
    is_allocated = false;
    data_size = data_sz;
    data = ptr;
  }

  void setup(int_t size, int_t data_sz) {
    is_allocated = true;
    data_size = data_sz;  
    data = malloc(size * data_size);
  }
  
  void destroy() {
    if (is_allocated) free(data);
    is_allocated = false;
    data_size = 0;
    data = nullptr;
  }
};


//==============================================================================
struct contra_mpi_task_info_t {
  std::map<contra_index_space_t*, contra_mpi_partition_t*> IndexPartMap;
  std::vector<contra_mpi_partition_t*> PartsToDelete;

  void register_partition(
      contra_index_space_t * is,
      contra_mpi_partition_t * part)
  { IndexPartMap.emplace(is, part); }

  std::pair<contra_mpi_partition_t*, bool>
    getOrCreatePartition(contra_index_space_t * is)
  {
    auto it = IndexPartMap.find(is);
    if (it != IndexPartMap.end()) {
      return {it->second, true};
    }
    else {
      auto part = new contra_mpi_partition_t;
      PartsToDelete.push_back(part);
      return {part, false};
    }
  }

  ~contra_mpi_task_info_t() {
    for (auto part : PartsToDelete)
      part->destroy();
  }
};


////////////////////////////////////////////////////////////////////////////////
// Function prototypes for mpi runtime
////////////////////////////////////////////////////////////////////////////////

/// index space creation
void contra_mpi_partition_from_size(
    int_t size,
    contra_index_space_t * is,
    contra_mpi_partition_t * part);

//void contra_mpi_init(int * argc, char *** argv);
//void contra_mpi_finalize();

} // extern


#endif // LIBRT_MPI_RT_HPP
