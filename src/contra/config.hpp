#ifndef CONFIG_HPP
#define CONFIG_HPP

//#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <cstdint>

namespace {

  using int_t = std::int64_t;
  using uint_t = std::uint64_t;
  using real_t = double;


  llvm::Type* llvmIntegerType( llvm::LLVMContext & TheContext )
  { return llvm::Type::getInt64Ty(TheContext); }
  
  llvm::Value* llvmInteger( llvm::LLVMContext & TheContext, int_t Val )
  { return llvm::ConstantInt::get(TheContext, llvm::APInt(64, Val, true)); }
  
  llvm::Type* llvmRealType( llvm::LLVMContext & TheContext )
  { return llvm::Type::getDoubleTy(TheContext); }
  
  llvm::Value* llvmReal( llvm::LLVMContext & TheContext, real_t Val )
  { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
  
  
  llvm::Type* llvmVoidPointerType( llvm::LLVMContext & TheContext )
  { return llvm::PointerType::get(llvm::Type::getInt8Ty(TheContext), 0); }

} // namespace

#endif // CONFIG_HPP
