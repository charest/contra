#ifndef LLVM_UTILS_HPP
#define LLVM_UTILS_HPP

#include "config.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <type_traits>
#include <vector>

namespace utils {

//============================================================================
// Helper structs to convert between LLVM and our types

template<typename T, typename Enable = void>
struct LlvmType;

template<typename T>
struct LlvmType<
  T, typename std::enable_if_t<std::is_same<T,void>::value> >
{
  static llvm::Type* getType(llvm::LLVMContext & TheContext)
  { return llvm::Type::getVoidTy(TheContext); }
};

template<typename T>
struct LlvmType<
  T, typename std::enable_if_t<std::is_same<T,void*>::value> >
{
  static llvm::Type* getType(llvm::LLVMContext & TheContext)
  { return llvm::PointerType::get(llvm::Type::getInt8Ty(TheContext), 0); }
};

template<typename T>
struct LlvmType< T,
  typename std::enable_if_t<(std::is_integral<T>::value || std::is_enum<T>::value)> >
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

//============================================================================
// Utility functions to get LLVM types and values

template<typename T>
auto llvmValue( llvm::LLVMContext & TheContext, T Val )
{ return LlvmType<T>::getValue(TheContext, Val); }

template<typename T>
llvm::Constant* llvmValue( llvm::LLVMContext & TheContext, llvm::Type* Ty, T Val )
{
  auto Size = sizeof(T) * 8;
  auto IsSigned = std::is_signed<T>::value;
  return llvm::Constant::getIntegerValue(Ty, llvm::APInt(Size, Val, IsSigned));
}

template<typename T>
auto llvmType( llvm::LLVMContext & TheContext )
{ return LlvmType<T>::getType(TheContext); }

std::vector<llvm::Type*> llvmTypes(const std::vector<llvm::Value*> & Vals);

//============================================================================  
// create a string
llvm::Value* llvmString(llvm::LLVMContext & TheContext,
    llvm::Module &TheModule, const std::string & Str);


//============================================================================  
// create a temporary builder
llvm::IRBuilder<> createBuilder(llvm::Function *TheFunction);

//============================================================================  
// create an entry block alloca
llvm::AllocaInst* createEntryBlockAlloca(llvm::Function *TheFunction,
  llvm::Type* Ty, const std::string & Name = "");

//============================================================================  
// get a types size
llvm::Value* getTypeSize(llvm::IRBuilder<> & Builder, llvm::Type* ElementType,
    llvm::Type* ResultType );

template<typename T>
llvm::Value* getTypeSize(llvm::IRBuilder<> & Builder, llvm::Type* ElementType)
{
  auto & TheContext = Builder.getContext();
  return getTypeSize(Builder, ElementType, llvmType<T>(TheContext));
}

} // namespace

#endif // CONFIG_HPP
