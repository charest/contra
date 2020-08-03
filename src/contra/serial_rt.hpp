#ifndef CONTRA_SERIAL_RT_HPP
#define CONTRA_SERIAL_RT_HPP

#include "config.hpp"
#include "tasking_rt.hpp"

#include "librt/dopevector.hpp"

#include <iostream>
#include <map>
#include <vector>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Serial runtime
////////////////////////////////////////////////////////////////////////////////

} // namespace

extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Types needed for serial runtime
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
struct contra_serial_partition_t {
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
  
};


//==============================================================================
struct contra_serial_field_t {
  int_t data_size;
  void *data;
  contra_index_space_t *index_space;

  void setup(
      contra_index_space_t *is,
      int_t data_sz)
  {
    auto size = is->size();
    data_size = data_sz;
    data = malloc(data_size*size);
    index_space = is;
  }

  void destroy() {
    free(data);
    data_size = 0;
    data = nullptr;
    index_space = nullptr;
  }
};

//==============================================================================
struct contra_serial_accessor_t {
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
struct contra_serial_partition_info_t {
  std::map<contra_index_space_t*, contra_serial_partition_t*> IndexPartMap;
  std::vector<contra_serial_partition_t*> PartsToDelete;

  void register_partition(
      contra_index_space_t * is,
      contra_serial_partition_t * part)
  { IndexPartMap.emplace(is, part); }

  std::pair<contra_serial_partition_t*, bool>
    getOrCreatePartition(contra_index_space_t * is)
  {
    auto it = IndexPartMap.find(is);
    if (it != IndexPartMap.end()) {
      return {it->second, true};
    }
    else {
      auto part = new contra_serial_partition_t;
      PartsToDelete.push_back(part);
      return {part, false};
    }
  }
  
  ~contra_serial_partition_info_t() {
    for (auto part : PartsToDelete)
      part->destroy();
  }
};


////////////////////////////////////////////////////////////////////////////////
// Function prototypes for serial runtime
////////////////////////////////////////////////////////////////////////////////

/// create partition info
void contra_serial_partition_info_create(contra_serial_partition_info_t**);
// destroy partition info
void contra_serial_partition_info_destroy(contra_serial_partition_info_t**);

/// create a field
void contra_serial_field_create(
    const char * name,
    int_t data_size,
    const void* init,
    contra_index_space_t * is,
    contra_serial_field_t * fld);
/// destroy a field
void contra_serial_field_destroy(contra_serial_field_t * fld);

/// index space partitioning
void contra_serial_index_space_create_from_partition(
    int_t i,
    contra_serial_partition_t * part,
    contra_index_space_t * is);

/// index space creation
void contra_legion_partition_from_index_space(
    contra_index_space_t * cs,
    contra_index_space_t * is,
    contra_serial_partition_t * part);

/// index space creation
void contra_serial_partition_from_size(
    int_t size,
    contra_index_space_t * is,
    contra_serial_partition_t * part);

/// index space creation
void contra_serial_partition_from_array(
    dopevector_t *arr,
    contra_index_space_t * is,
    contra_serial_partition_t * part);

/// destroy a partition
void contra_serial_partition_destroy(contra_serial_partition_t * part);

} // extern


#endif // LIBRT_LEGION_RT_HPP
