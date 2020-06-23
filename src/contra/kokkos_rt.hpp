#ifndef CONTRA_KOKKOS_RT_HPP
#define CONTRA_KOKKOS_RT_HPP

#include "config.hpp"
#include "tasking_rt.hpp"

#include <Kokkos_Core.hpp>

#include <memory>
#include <vector>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Kokkos runtime
////////////////////////////////////////////////////////////////////////////////

struct KokkosRuntime {
  Kokkos::ScopeGuard kokkos;
  KokkosRuntime(int argc, char * argv[]) : kokkos(argc, argv) {}
};

using RangePolicy = Kokkos::RangePolicy<
  Kokkos::DefaultExecutionSpace,
  Kokkos::IndexType<int_t>>;

using KokkosIntField = Kokkos::View<int_t*>;
using KokkosRealField = Kokkos::View<real_t*>;

enum KokkosFieldType {
  Integer,
  Real
};

} // namespace

extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Types needed for kokkos runtime
////////////////////////////////////////////////////////////////////////////////


//==============================================================================
struct contra_kokkos_field_t {
  int data_type;
  void *field;
};

//==============================================================================
struct contra_kokkos_task_t {
  std::vector<byte_t> data;
};

////////////////////////////////////////////////////////////////////////////////
// Function prototypes for kokkos runtime
////////////////////////////////////////////////////////////////////////////////

/// start the runtime
int contra_kokkos_runtime_start(int, char **);
/// stop the runtime
void contra_kokkos_runtime_stop();

/// create a field
void contra_kokkos_field_create(
    const char * name,
    int data_type,
    const void* init,
    contra_index_space_t * is,
    contra_kokkos_field_t * fld);
/// destroy a field
void contra_kokkos_field_destroy(contra_kokkos_field_t * fld);
  
/// create task data
byte_t* contra_kokkos_task_create(contra_kokkos_task_t ** task, int_t size);
/// destroy task data
void contra_kokkos_task_destroy(contra_kokkos_task_t ** task);

} // extern


#endif // LIBRT_LEGION_RT_HPP
