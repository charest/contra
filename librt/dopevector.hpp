#ifndef RTLIB_DOPEVECTOR_HPP
#define RTLIB_DOPEVECTOR_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

extern "C" {

/// forward declaration
struct dopevector_t;
struct int_t;

/// memory allocation
DLLEXPORT dopevector_t allocate(int_t size);

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
