#ifndef RTLIB_MATH_HPP
#define RTLIB_MATH_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

extern "C" {

DLLEXPORT double mysqrt(double);
DLLEXPORT double myabs(double);

} // extern

namespace librt {

llvm::Function *installCSqrt(llvm::LLVMContext &, llvm::Module &);
llvm::Function *installCAbs(llvm::LLVMContext &, llvm::Module &);
llvm::Function *installCMax(llvm::LLVMContext &, llvm::Module &);

llvm::Function *installSqrt(llvm::LLVMContext &, llvm::Module &);
llvm::Function *installAbs(llvm::LLVMContext &, llvm::Module &);

} // namespace


#endif // RTLIB_MATH_HPP
