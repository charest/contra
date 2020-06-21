#ifndef CONTRA_KOKKOS_RT_HPP
#define CONTRA_KOKKOS_RT_HPP


////////////////////////////////////////////////////////////////////////////////
// Create new legion Reduction op
////////////////////////////////////////////////////////////////////////////////

namespace contra {


/// start the runtime
int contra_kokkos_runtime_start(int, char **);
/// stop the runtime
void contra_kokkos_runtime_stop();

} // extern


#endif // LIBRT_LEGION_RT_HPP
