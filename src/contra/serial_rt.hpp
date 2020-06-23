#ifndef CONTRA_SERIAL_RT_HPP
#define CONTRA_SERIAL_RT_HPP

#include "config.hpp"
#include "tasking_rt.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Serial runtime
////////////////////////////////////////////////////////////////////////////////

} // namespace

extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Types needed for kokkos runtime
////////////////////////////////////////////////////////////////////////////////


//==============================================================================
struct contra_serial_field_t {
  int_t data_size;
  void *data;
};


////////////////////////////////////////////////////////////////////////////////
// Function prototypes for kokkos runtime
////////////////////////////////////////////////////////////////////////////////

/// create a field
void contra_serial_field_create(
    const char * name,
    int_t data_size,
    const void* init,
    contra_index_space_t * is,
    contra_serial_field_t * fld);
/// destroy a field
void contra_serial_field_destroy(contra_serial_field_t * fld);

} // extern


#endif // LIBRT_LEGION_RT_HPP
