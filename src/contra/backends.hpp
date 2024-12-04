#ifndef CONTRA_BACKENDS_HPP
#define CONTRA_BACKENDS_HPP

#include "config.hpp"
#include "utils/string_utils.hpp"

namespace contra {

enum class SupportedBackends {
  Serial,
#ifdef HAVE_LEGION
  Legion,
#endif
#ifdef HAVE_CUDA
  Cuda,
#endif
#ifdef HAVE_ROCM
  ROCm,
#endif
#ifdef HAVE_THREADS
  Threads,
#endif
#ifdef HAVE_MPI
  MPI,
#endif
  Size
};

inline SupportedBackends getBackend(const std::string & Name)
{
  auto lower = utils::tolower(Name);
#ifdef HAVE_LEGION
  if (lower == "legion") return SupportedBackends::Legion; 
#endif
#ifdef HAVE_CUDA
  if (lower == "cuda") return SupportedBackends::Cuda;
#endif
#ifdef HAVE_ROCM
  if (lower == "rocm") return SupportedBackends::ROCm;
#endif
#ifdef HAVE_THREADS
  if (lower == "threads") return SupportedBackends::Threads;
#endif
#ifdef HAVE_MPI
  if (lower == "mpi") return SupportedBackends::MPI;
#endif
  if (lower == "serial") return SupportedBackends::Serial;
  return SupportedBackends::Size;
}

} // namespace

#endif // CONTRA_BACKENDS_HPP
