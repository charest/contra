#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <cstdint>

namespace {

  using int_t = std::int64_t;
  using uint_t = std::uint64_t;
  using real_t = double;


  llvm::Type* llvmIntegerType( llvm::LLVMContext & TheContext )
  { return llvm::Type::getInt64Ty(TheContext); }
  
  llvm::Type* llvmRealType( llvm::LLVMContext & TheContext )
  { return llvm::Type::getDoubleTy(TheContext); }
  
  llvm::Type* llvmVoidPointerType( llvm::LLVMContext & TheContext )
  { return llvm::PointerType::get(llvm::Type::getInt8Ty(TheContext), 0); }

} // namespace

#endif // CONFIG_HPP
