#ifndef RTLIB_DOPEVECTOR_HPP
#define RTLIB_DOPEVECTOR_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

#include <cstdint>

extern "C" {

/// simple dopevector type
struct dopevector_t {
  void * data = nullptr;
  std::uint64_t size = 0;
};

/// memory allocation
DLLEXPORT dopevector_t allocate(std::uint64_t size);

/// memory deallocation
DLLEXPORT void deallocate(dopevector_t dv);

} // extern

namespace librt {

// install memory allocator
llvm::Function *installAllocate(llvm::LLVMContext &, llvm::Module &);
// install memory deallocator
llvm::Function *installDeAllocate(llvm::LLVMContext &, llvm::Module &);

} // namespace


#endif // RTLIB_DOPEVECTOR_HPP
