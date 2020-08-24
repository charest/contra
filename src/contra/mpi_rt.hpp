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

////////////////////////////////////////////////////////////////////////////////
/// field registry data
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
/// mpi runtime 
////////////////////////////////////////////////////////////////////////////////
struct field_exchange_t {
  std::vector<void*> RecvBufs;
  std::vector<MPI_Request> Requests;
  std::vector<int_t> Locations;
  

  void setup(int_t recvsize, int_t reqsize)
  {
    RecvBufs.emplace_back( malloc(recvsize) );
    Requests.reserve(reqsize);
  }
  
  void setup(int_t recvsize, int_t sendsize, int_t reqsize)
  {
    RecvBufs.emplace_back( malloc(recvsize) );
    RecvBufs.emplace_back( malloc(sendsize) );
    Requests.reserve(reqsize);
  }

  auto getBuffer(int i=0) const { return RecvBufs[i]; }
  auto & getRequests() { return Requests; }

  auto transferBuffer(int i=0) {
    auto buf = RecvBufs[i];
    RecvBufs[i] = nullptr;
    return buf;
  }

  ~field_exchange_t() {
    for (auto RecvBuf : RecvBufs)
      if (RecvBuf) free(RecvBuf);
  }
};

////////////////////////////////////////////////////////////////////////////////
/// mpi runtime
////////////////////////////////////////////////////////////////////////////////
class mpi_runtime_t {

  int Rank = -1;
  int Size = 0;
  unsigned TaskCounter = 0;
  unsigned FieldCounter = 0;
  unsigned PartitionCounter = 0;
  
  std::map<unsigned, field_registry_t> FieldRegistry;
  std::map<void*, field_exchange_t> FieldRequests;

  std::map<unsigned, unsigned> PartitionRegistry;


public:
  
  void setup(int rank, int size)
  {
    Rank = rank;
    Size = size;
  }

  void incrementTaskCounter() { TaskCounter++; }
  void decrementTaskCounter() { TaskCounter--; }
  auto getTaskCounter() { return TaskCounter; }

  void check(int);
  bool isRoot() const { return Rank == 0; }

  auto getSize() const { return Size; }
  auto getRank() const { return Rank; }

  auto registerField(int_t data_size, const void * init)
  {
    auto fid = FieldCounter++;
    FieldRegistry.emplace(fid, field_registry_t{data_size, init});
    return fid;
  }
  void deregisterField(unsigned i) { FieldRegistry.erase(i); }

  auto & getRegisteredField(unsigned i) { return FieldRegistry.at(i); }

  auto & requestField(void * key, int_t recvcnt, int_t reqcnt) 
  { 
    auto & obj = FieldRequests[key];
    obj.setup(recvcnt, reqcnt);
    return obj;
  }
  
  auto & requestField(void * key, int_t recvcnt, int_t sendcnt, int_t reqcnt) 
  { 
    auto & obj = FieldRequests[key];
    obj.setup(recvcnt, sendcnt, reqcnt);
    return obj;
  }


  std::pair<field_exchange_t*, bool> findFieldRequest(void* data)
  {
    auto it = FieldRequests.find(data);
    if (it != FieldRequests.end()) {
      return {&it->second, true};
    }
    else {
      return {nullptr, false};
    }
  }

  void eraseFieldRequest(void *data)
  { FieldRequests.erase(data); }

  auto registerPartition() {
    auto pid = PartitionCounter++;
    PartitionRegistry.emplace(pid, 1);
    return pid;
  }

  auto decrementPartition(unsigned id) {
    auto & Part = PartitionRegistry.at(id);
    if (Part <= 1) {
      PartitionRegistry.erase(id);
      return true;
    }
    else {
      Part--;
      return false;
    }
  }
  
  void incrementPartition(unsigned id)
  { 
    PartitionRegistry[id]++;
  }

};

} // namespace


extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Types needed for mpi runtime
////////////////////////////////////////////////////////////////////////////////

struct contra_mpi_field_t;

//==============================================================================
struct contra_mpi_partition_t {
  int_t part_size;
  int_t num_parts;
  int_t* offsets;
  contra_mpi_field_t* indices;
  contra_index_space_t *index_space;
  int id;

  void setup(
      int_t part_sz,
      int_t parts,
      contra_index_space_t *is,
      int_t * offs,
      int pid)
  {
    num_parts = parts;
    part_size = part_sz;
    index_space = is;
    offsets = offs;
    indices = nullptr;
    id = pid;
  }

  void setup(
      int_t part_sz,
      int_t parts,
      contra_index_space_t *is,
      contra_mpi_field_t * indx,
      int_t * offs,
      int pid)
  {
    num_parts = parts;
    part_size = part_sz;
    index_space = is;
    offsets = offs;
    indices = indx;
    id = pid;
  }

  void destroy() {
    if (offsets) {
      delete[] offsets;
      offsets = nullptr;
    }
    indices = nullptr;
    part_size = 0;
    num_parts = 0;
    index_space = nullptr;
  }

  auto size(int_t i) { return offsets[i+1] - offsets[i]; }

  auto offsets_begin() { return offsets; }
  auto offsets_end() { return offsets + num_parts + 1; }
  
};


//==============================================================================
struct contra_mpi_field_t {
  int_t data_size = 0;
  void *data = nullptr;
  contra_index_space_t *index_space = nullptr;
  int id = -1;
  int_t * distribution = nullptr;
  contra_mpi_partition_t * partition = nullptr;

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
    partition = nullptr;
  }

  void destroy() {
    if (data) free(data);
    data_size = 0;
    data = nullptr;
    index_space = nullptr;
    id = -1;
    if (distribution) delete[] distribution;
    if (partition) delete partition;
  }

  int_t allocate(contra_mpi_partition_t *part, int_t *dist, int_t rank, int_t size)
  {
    distribution = new int_t[size+1];
    memcpy(distribution, dist, (size+1)*sizeof(int_t));

    partition = new contra_mpi_partition_t;
    *partition = *part;
    
    auto len = part->offsets[dist[rank+1]] - part->offsets[dist[rank]];
    data = malloc(data_size*len);
    return len;
  }

  void redistribute(contra_mpi_partition_t *part, int_t *dist, int_t size)
  {
    *partition = *part;
    memcpy(distribution, dist, (size+1)*sizeof(int_t));
  }

  void transfer(void * buf) {
    if (data) free(data);
    data = buf;
  }
  
  int_t rank_begin(int_t i) { return partition->offsets[distribution[i]]; }
  int_t rank_end(int_t i) { return partition->offsets[distribution[i+1]]; }


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


void contra_mpi_partition_destroy(contra_mpi_partition_t * part);

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
      contra_mpi_partition_destroy(part);
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
