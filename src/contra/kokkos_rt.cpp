#include "kokkos_rt.hpp"

#include <sstream>

using namespace contra;

std::unique_ptr<KokkosRuntime> KokkosRT;

extern "C" {
  
//==============================================================================
// start the runtime
//==============================================================================
int contra_kokkos_runtime_start(int argc, char * argv[])
{ 
  KokkosRT = std::make_unique<KokkosRuntime>(argc, argv);
  return 0;
}

//==============================================================================
// stop the runtime
//==============================================================================
void contra_kokkos_runtime_stop() {}


//==============================================================================
// Create a field
//==============================================================================
void contra_kokkos_field_create(
    const char * name,
    int data_type,
    const void* init,
    contra_index_space_t * is,
    contra_kokkos_field_t * fld)
{
  auto size = is->end - is->start;
  
  std::stringstream ss;
  ss << "__init_" << name << "__";

  //------------------------------------
  if (data_type == KokkosFieldType::Real) {
    auto view_ptr = new KokkosRealField(std::string(name), size);
    if (init) {
      auto init_as_real = *static_cast<const real_t*>(init);
      auto & view =  *view_ptr;
      Kokkos::parallel_for(
          ss.str(),
          RangePolicy(0, size),
          KOKKOS_LAMBDA (int_t i) {
            view(i) = init_as_real;
          });
    }
    fld->field = static_cast<void*>(view_ptr);
  }
  //------------------------------------
  else if (data_type == KokkosFieldType::Integer) {
    auto view_ptr = new KokkosIntField(std::string(name), size);
    if (init) {
      auto init_as_int = *static_cast<const int_t*>(init);
      auto & view =  *view_ptr;
      Kokkos::parallel_for(
          ss.str(),
          RangePolicy(0, size),
          KOKKOS_LAMBDA (int_t i) {
            view(i) = init_as_int;
          });
    }
    fld->field = static_cast<void*>(view_ptr);
  }
  //------------------------------------
  else {
    std::cerr 
      << "Unknown data type!  Only reals and integers are supported "
      << "right now."
      << std::endl;
    abort();
  }

  fld->data_type = data_type;
}

//==============================================================================
// Destroy a field
//==============================================================================
void contra_kokkos_field_destroy(contra_kokkos_field_t * fld)
{
  if (fld->data_type == KokkosFieldType::Real) {
    auto view_ptr = static_cast<KokkosRealField*>(fld->field);
    delete view_ptr;
  }
  else if (fld->data_type == KokkosFieldType::Integer) {
    auto view_ptr = static_cast<KokkosIntField*>(fld->field);
    delete view_ptr;
  }
  else {
    std::cerr 
      << "Unknown data type!  Only reals and integers are supported "
      << "right now."
      << std::endl;
    abort();
  }
  fld->data_type = 0;
  fld->field = nullptr;
}

//==============================================================================
/// create task data
//==============================================================================
byte_t * contra_kokkos_task_create(contra_kokkos_task_t ** task, int_t size)
{
  *task = new contra_kokkos_task_t;
  (*task)->data.resize(size);
  return (*task)->data.data();
}

//==============================================================================
/// destroy task data
//==============================================================================
void contra_kokkos_task_destroy(contra_kokkos_task_t ** task)
{ delete *task; }

} // extern
