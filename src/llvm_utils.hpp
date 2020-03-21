#ifndef LLVM_UTILS_HPP
#define LLVM_UTILS_HPP

#include "config.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <type_traits>

namespace {

  template<typename T = int_t>
  llvm::Type* llvmIntegerType( llvm::LLVMContext & TheContext )
  {
    switch (sizeof(T)) {
      case  1: return llvm::Type::getInt8Ty(TheContext);
      case  2: return llvm::Type::getInt16Ty(TheContext);
      case  4: return llvm::Type::getInt32Ty(TheContext);
      case  8: return llvm::Type::getInt64Ty(TheContext);
      case 16: return llvm::Type::getInt128Ty(TheContext);
    };
    return nullptr;
  }
  
  template<typename T>
  llvm::Value* llvmInteger( llvm::LLVMContext & TheContext, T Val,
      bool IsSigned=std::is_signed<T>::value )
  {
    auto IntType = llvmIntegerType<T>(TheContext);
    auto Size = IntType->getIntegerBitWidth();
    return llvm::ConstantInt::get(TheContext, llvm::APInt(Size, Val, IsSigned));
  }
  
  inline llvm::Type* llvmRealType( llvm::LLVMContext & TheContext )
  { return llvm::Type::getDoubleTy(TheContext); }
  
  inline llvm::Value* llvmReal( llvm::LLVMContext & TheContext, real_t Val )
  { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
  
  
  inline llvm::Type* llvmVoidPointerType( llvm::LLVMContext & TheContext )
  { return llvm::PointerType::get(llvm::Type::getInt8Ty(TheContext), 0); }

  inline llvm::Value* llvmString(llvm::LLVMContext & TheContext,
      llvm::Module &TheModule, const std::string & Str)
  {
    using namespace llvm;
    auto ConstantArray = ConstantDataArray::getString(TheContext, Str);
    auto GVStr = new GlobalVariable(TheModule, ConstantArray->getType(), true,
        GlobalValue::InternalLinkage, ConstantArray);
    auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
    auto StrV = ConstantExpr::getGetElementPtr(
        IntegerType::getInt8Ty(TheContext), GVStr, ZeroC, true);
    return StrV;
  }

  //============================================================================
  template<typename T>
  llvm::Type* llvmType( llvm::LLVMContext & TheContext );
  
  template<>
  llvm::Type* llvmType<void*>( llvm::LLVMContext & TheContext )
  { return llvm::PointerType::get(llvm::Type::getInt8Ty(TheContext), 0); }
  
  template<>
  llvm::Type* llvmType<bool>( llvm::LLVMContext & TheContext )
  { return llvm::Type::getInt1Ty(TheContext); }

  template<>
  llvm::Type* llvmType<float>( llvm::LLVMContext & TheContext )
  { return llvm::Type::getFloatTy(TheContext); }

  template<>
  llvm::Type* llvmType<double>( llvm::LLVMContext & TheContext )
  { return llvm::Type::getDoubleTy(TheContext); }
  
  template<typename T>
  std::enable_if_t<(std::is_integral<T>::value && !std::is_same<T,bool>::value), llvm::Type*>
  llvmType( llvm::LLVMContext & TheContext )
  {
    switch (sizeof(T)) {
      case  1: return llvm::Type::getInt8Ty(TheContext);
      case  2: return llvm::Type::getInt16Ty(TheContext);
      case  4: return llvm::Type::getInt32Ty(TheContext);
      case  8: return llvm::Type::getInt64Ty(TheContext);
      case 16: return llvm::Type::getInt128Ty(TheContext);
    };
    return nullptr;
  }

  //============================================================================
  template<typename T>
  llvm::Value* llvmValue( llvm::LLVMContext & TheContext, T Val);

  template<>
  llvm::Value* llvmValue<bool>( llvm::LLVMContext & TheContext, bool Val)
  { return llvm::ConstantInt::get(TheContext, llvm::APInt(1, Val)); }
  
  template<typename T>
  std::enable_if_t<std::is_floating_point<T>::value, llvm::Value*>
  llvmValue( llvm::LLVMContext & TheContext, T Val )
  { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
  
  template<typename T>
  std::enable_if_t<(std::is_integral<T>::value && !std::is_same<T,bool>::value), llvm::Value*>
  llvmType( llvm::LLVMContext & TheContext, T Val )
  {
    auto IntType = llvmType<T>(TheContext);
    auto Size = IntType->getIntegerBitWidth();
    return llvm::ConstantInt::get(TheContext, llvm::APInt(Size, Val,
          std::is_signed<T>::value));
  }

  llvm::Value* llvmValue(llvm::LLVMContext & TheContext,
      llvm::Module &TheModule, const std::string & Str)
  {
    using namespace llvm;
    auto ConstantArray = ConstantDataArray::getString(TheContext, Str);
    auto GVStr = new GlobalVariable(TheModule, ConstantArray->getType(), true,
        GlobalValue::InternalLinkage, ConstantArray);
    auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
    auto StrV = ConstantExpr::getGetElementPtr(
        IntegerType::getInt8Ty(TheContext), GVStr, ZeroC, true);
    return StrV;
  }

} // namespace

#endif // CONFIG_HPP
