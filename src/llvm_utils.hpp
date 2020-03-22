#ifndef LLVM_UTILS_HPP
#define LLVM_UTILS_HPP

#include "config.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <type_traits>

namespace {

  //============================================================================
  
  template<typename T, typename Enable = void>
  struct LlvmType;
  
  template<typename T>
  struct LlvmType<
    T, typename std::enable_if_t<std::is_same<T,void*>::value> >
  {
    static llvm::Type* getType(llvm::LLVMContext & TheContext)
    { return llvm::PointerType::get(llvm::Type::getInt8Ty(TheContext), 0); }
  };

  template<typename T>
  struct LlvmType<
    T, typename std::enable_if_t<std::is_same<T,bool>::value> >
  {
    static llvm::Type* getType(llvm::LLVMContext & TheContext)
    {return llvm::Type::getInt1Ty(TheContext); }

    static llvm::Value* getValue(llvm::LLVMContext & TheContext, T Val)
    { return llvm::ConstantInt::get(TheContext, llvm::APInt(1, Val)); }
  };

  template<typename T>
  struct LlvmType< T,
    typename std::enable_if_t<(std::is_integral<T>::value || std::is_enum<T>::value) 
      && !std::is_same<T,bool>::value> >
  {
    static llvm::Type* getType(llvm::LLVMContext & TheContext)
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
    
    static llvm::Value* getValue(llvm::LLVMContext & TheContext, T Val)
    {
      auto IntType = getType(TheContext);
      auto Size = IntType->getIntegerBitWidth();
      auto IsSigned = std::is_signed<T>::value;
      return llvm::ConstantInt::get(TheContext, llvm::APInt(Size, Val, IsSigned));
    }
  };

  template<typename T>
  struct LlvmType<
    T, typename std::enable_if_t<std::is_same<T,float>::value> >
  {
    static llvm::Type* getType(llvm::LLVMContext & TheContext)
    {return llvm::Type::getFloatTy(TheContext); }

    static llvm::Value* getValue(llvm::LLVMContext & TheContext, T Val)
    { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
  };

  template<typename T>
  struct LlvmType<
    T, typename std::enable_if_t<std::is_same<T,double>::value> >
  {
    static llvm::Type* getType(llvm::LLVMContext & TheContext)
    {return llvm::Type::getDoubleTy(TheContext); }
    
    static llvm::Value* getValue(llvm::LLVMContext & TheContext, T Val)
    { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
  };

  template<typename T>
  llvm::Type* llvmType( llvm::LLVMContext & TheContext )
  { return LlvmType<T>::getType(TheContext); }
  
  template<typename T>
  llvm::Value* llvmValue( llvm::LLVMContext & TheContext, T Val )
  { return LlvmType<T>::getValue(TheContext, Val); }
  
  //============================================================================
  
  
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

} // namespace

#endif // CONFIG_HPP
