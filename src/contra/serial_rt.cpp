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
  is->setup(part->offsets[i], part->offsets[i+1]);
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

  part->setup( index_size, num_parts, is, offsets);
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

    part->setup( index_size, num_parts, is, offsets);

  }
  //------------------------------------
}

//==============================================================================
/// index space creation
//==============================================================================
void contra_serial_partition_from_field(
    contra_serial_field_t *fld,
    contra_index_space_t * is,
    contra_serial_partition_t * fld_part,
    contra_serial_partition_t * part)
{
  auto num_parts = fld_part->num_parts;

  auto indices = new int_t*[num_parts];
  auto offsets = new int_t[num_parts+1];
  offsets[0] = 0;
  
  auto fld_ptr = static_cast<int_t*>(fld->data);

  if (auto fld_indices = fld_part->indices) {

    for (int_t i=0; i<num_parts; ++i) {
      auto size = fld_part->size(i);
      offsets[i+1] = offsets[i] + size;
      auto & part_indices = indices[i];
      part_indices = new int_t[size];
      auto fld_part_indices = fld_indices[i];
      for (int_t j=0; j<size; ++j) {
        part_indices[j] = fld_ptr[ fld_part_indices[j] ]; 
      }
    }

  }
  else {
    
    auto fld_offsets = fld_part->offsets;

    for (int_t i=0; i<num_parts; ++i) {
      auto size = fld_part->size(i);
      offsets[i+1] = offsets[i] + size;
      auto & part_indices = indices[i];
      part_indices = new int_t[size];
      auto fld_part_offset = fld_offsets[i];
      for (int_t j=0; j<size; ++j) {
        part_indices[j] = fld_ptr[ fld_part_offset + j ]; 
      }
    }

  }

  auto expanded_size = fld->index_space->size();
  part->setup(expanded_size, num_parts, is, indices, offsets);
}

//==============================================================================
// Destroy a partition
//==============================================================================
void contra_serial_partition_destroy(contra_serial_partition_t * part)
{ part->destroy(); }

//==============================================================================
/// Accessor write
//==============================================================================
contra_serial_partition_t*  contra_serial_partition_get(
    contra_index_space_t * cs,
    contra_serial_field_t * fld,
    contra_serial_partition_info_t **info)
{
  // no partitioning specified
  auto res = (*info)->getOrCreatePartition(fld->index_space);
  auto part = res.first;
  if (!res.second) {
    contra_serial_partition_from_index_space(
        cs,
        fld->index_space,
        part);
    (*info)->register_partition(fld->index_space, part);
  }

  return part;
}

//==============================================================================
/// Set an accessors current partition.
//==============================================================================
void contra_serial_accessor_setup(
    int_t i,
    contra_serial_partition_t * part,
    contra_serial_field_t * fld,
    contra_serial_accessor_t * acc)
{
  auto fld_data = static_cast<byte_t*>(fld->data);
  auto data_size = fld->data_size;

  if (part->indices) {
    auto size = part->size(i);
    acc->setup( size, data_size );
    auto dest = static_cast<byte_t*>(acc->data);
    auto indices = part->indices[i];
    for (int_t j=0; j<size; ++j) {
      const auto src = fld_data + data_size*indices[j];
      memcpy(dest, src, data_size);
      dest += data_size;
    }
  }
  else {
    auto offsets = part->offsets;
    acc->setup( fld_data + data_size*offsets[i], data_size );
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
