#ifndef RTLIB_PRINT_HPP
#define RTLIB_PRINT_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

extern "C" {

/// generic c print statement
DLLEXPORT void print(const char *format, ...);

} // extern

namespace librt {

// install print statement
llvm::Function *installPrint(llvm::LLVMContext &, llvm::Module &);

} // namespace


#endif // RTLIB_PRINT_HPP
