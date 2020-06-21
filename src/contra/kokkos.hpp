#ifndef CONTRA_KOKKOS_HPP
#define CONTRA_KOKKOS_HPP

#include "config.hpp"

#ifdef HAVE_KOKKOS

#include "tasking.hpp"

namespace contra {

class KokkosTasker : public AbstractTasker {

public:
 
  KokkosTasker(utils::BuilderHelper & TheHelper);

  virtual llvm::Value* startRuntime(
      llvm::Module &,
      int,
      char **) override;
  virtual void stopRuntime(llvm::Module &) override;
};

} // namepsace

#endif // HAVE_KOKKOS
#endif // LIBRT_LEGION_HPP
