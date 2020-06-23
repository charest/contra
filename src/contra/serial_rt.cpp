#include "serial_rt.hpp"

#include <cstring>
#include <cstdlib>

using namespace contra;

extern "C" {
  

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
  auto size = is->end - is->start;
  fld->data = malloc(data_size*size);
  fld->data_size = data_size;
  
  auto ptr = static_cast<byte_t*>(fld->data);
  for (int_t i=0; i<size; ++i)
    memcpy(ptr + i*data_size, init, data_size);
}

//==============================================================================
// Destroy a field
//==============================================================================
void contra_serial_field_destroy(contra_serial_field_t * fld)
{
  free(fld->data);
  fld->data_size = 0;
  fld->data = nullptr;
}

} // extern
