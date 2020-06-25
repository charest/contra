#include "serial_rt.hpp"

#include <cstring>
#include <cstdlib>
#include <iostream>

using namespace contra;

extern "C" {
  
//==============================================================================
/// create partition info
//==============================================================================
void contra_serial_partition_info_create(contra_serial_partition_info_t** info)
{ *info = new contra_serial_partition_info_t; }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_serial_partition_info_destroy(contra_serial_partition_info_t** info)
{ delete (*info); }

//==============================================================================
// destroy partition info
//==============================================================================
void contra_serial_register_index_partition(
    contra_index_space_t * is,
    contra_serial_partition_t * part,
    contra_serial_partition_info_t** info)
{
  (*info)->register_partition(is, part);
}

//==============================================================================
// Create a field
//==============================================================================
void contra_serial_field_create(
    const char * name,
    int_t data_size,
    const void* init,
    contra_index_space_t * is,
    contra_serial_field_t * fld)
{
  fld->setup(is, data_size);
  auto ptr = static_cast<byte_t*>(fld->data);
  for (int_t i=0; i<is->size(); ++i)
    memcpy(ptr + i*data_size, init, data_size);
}

//==============================================================================
// Destroy a field
//==============================================================================
void contra_serial_field_destroy(contra_serial_field_t * fld)
{ fld->destroy(); }

//==============================================================================
/// index space partitioning
//==============================================================================
void contra_serial_index_space_create_from_partition(
    int_t i,
    contra_serial_partition_t * part,
    contra_index_space_t * is)
{
  if (part->offsets) {
    is->setup(part->offsets[i], part->offsets[i+1]);
  }
  else if (part->indices) {
    abort();
  }
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_serial_partition_from_index_space(
    contra_index_space_t * cs,
    contra_index_space_t * is,
    contra_serial_partition_t * part)
{
  auto num_parts = cs->size();
  contra_serial_partition_from_size(
      num_parts,
      is,
      part);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_serial_partition_from_size(
    int_t num_parts,
    contra_index_space_t * is,
    contra_serial_partition_t * part)
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

  part->setup( index_size, index_size, num_parts, is, offsets);
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_serial_partition_from_array(
    dopevector_t *arr,
    contra_index_space_t * is,
    contra_serial_partition_t * part)
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

    part->setup( index_size, index_size, num_parts, is, offsets);

  }
  //------------------------------------
}

//==============================================================================
// Destroy a partition
//==============================================================================
void contra_serial_partition_destroy(contra_serial_partition_t * part)
{ part->destroy(); }

//==============================================================================
/// Accessor write
//==============================================================================
void contra_serial_accessor_create(
    contra_index_space_t * cs,
    contra_serial_partition_t * part,
    contra_serial_partition_info_t **info,
    contra_serial_field_t * fld,
    contra_serial_accessor_t * acc)
{
  // no partitioning specified
  if(!part) {
    auto res = (*info)->getOrCreatePartition(fld->index_space);
    part = res.first;
    if (!res.second) {
      contra_serial_partition_from_index_space(
          cs,
          fld->index_space,
          part);
      (*info)->register_partition(fld->index_space, part);
    }
  }

  acc->setup(part, fld);
}

//==============================================================================
/// Set an accessors current partition.
//==============================================================================
void contra_serial_accessor_set_current(
    int_t i,
    contra_serial_accessor_t * acc)
{
  if (auto offsets = acc->index_part->offsets) {
    acc->is_allocated = false;
    auto fld = acc->field;
    acc->data = static_cast<byte_t*>(fld->data) + acc->data_size*offsets[i];
  }
  else if (acc->index_part->indices) {
    abort();
  }
  else {
    std::cerr << "Offsets AND indices are not set!" << std::endl;
    abort();
  }
}

//==============================================================================
/// Accessor write
//==============================================================================
void contra_serial_accessor_write(
    contra_serial_accessor_t * acc,
    const void * data,
    int_t index)
{
  byte_t * offset = static_cast<byte_t*>(acc->data) + acc->data_size*index;
  memcpy(offset, data, acc->data_size);
}

//==============================================================================
/// Accessor read
//==============================================================================
void contra_serial_accessor_read(
    contra_serial_accessor_t * acc,
    void * data,
    int_t index)
{
  const byte_t * offset = static_cast<const byte_t*>(acc->data) + acc->data_size*index;
  memcpy(data, offset, acc->data_size);
}

//==============================================================================
/// Accessor read
//==============================================================================
void contra_serial_accessor_destroy(
    contra_serial_accessor_t * acc)
{ acc->destroy(); }


} // extern
