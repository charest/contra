#ifndef CONTRA_BACKENDS_HPP
#define CONTRA_BACKENDS_HPP

#include "config.hpp"
 
namespace contra {

enum class SupportedBackends {
#ifdef HAVE_LEGION
  Legion,
#endif
#ifdef HAVE_KOKKOS
  Kokkos,
#endif
  Size
};

} // namespace

#endif // CONTRA_BACKENDS_HPP
