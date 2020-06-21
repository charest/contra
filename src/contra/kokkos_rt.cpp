#include "kokkos_rt.hpp"

#include <Kokkos_Core.hpp>

extern "C" {
  
//==============================================================================
// start the runtime
//==============================================================================
int contra_kokkos_runtime_start(int argc, char * argv[])
{ 
  Kokkos::initialize(argc, argv);
  return 0;
}

//==============================================================================
// stop the runtime
//==============================================================================
void contra_kokkos_runtime_stop()
{ Kokkos::finalize(); }

} // extern
