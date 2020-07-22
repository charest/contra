#ifndef LLVM_UTILS_HPP
#define LLVM_UTILS_HPP

#include "config.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

namespace llvm {
class Target;
}

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
  
  static llvm::Constant* getValue(llvm::LLVMContext & TheContext, T Val)
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

  static llvm::Constant* getValue(llvm::LLVMContext & TheContext, T Val)
  { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
};

template<typename T>
struct LlvmType<
  T, typename std::enable_if_t<std::is_same<T,double>::value> >
{
  static llvm::Type* getType(llvm::LLVMContext & TheContext)
  {return llvm::Type::getDoubleTy(TheContext); }
  
  static llvm::Constant* getValue(llvm::LLVMContext & TheContext, T Val)
  { return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val)); }
};

//============================================================================
// Utility functions to get LLVM types and values

template<typename T>
auto llvmValue( llvm::LLVMContext & TheContext, T Val )
{ return LlvmType<T>::getValue(TheContext, Val); }

template<
  typename T,
  typename = typename std::enable_if_t<!std::is_floating_point<T>::value>
  >
llvm::Constant* llvmValue( llvm::LLVMContext & TheContext, llvm::Type* Ty, T Val )
{
  auto Size = Ty->getIntegerBitWidth();
  auto IsSigned = std::is_signed<T>::value;
  return llvm::Constant::getIntegerValue(Ty, llvm::APInt(Size, Val, IsSigned));
}

template<
  typename T,
  typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr
  >
llvm::Constant* llvmValue( llvm::LLVMContext & TheContext, llvm::Type* Ty, T Val )
{
  return llvm::ConstantFP::get(TheContext, llvm::APFloat(Val));
}

template<typename T>
auto llvmType( llvm::LLVMContext & TheContext )
{ return LlvmType<T>::getType(TheContext); }

std::vector<llvm::Type*> llvmTypes(const std::vector<llvm::Value*> & Vals);

//============================================================================  
// create a string
llvm::Constant* llvmString(
    llvm::LLVMContext & TheContext,
    llvm::Module &TheModule,
    const std::string & Str);

//============================================================================
llvm::Constant* llvmArray(
    llvm::LLVMContext & TheContext,
    llvm::Module &TheModule,
    const std::vector<llvm::Constant*> & ValsC,
    const std::vector<llvm::Constant*> & GEPIndices = {});

template<typename T, typename U = T>
llvm::Constant* llvmArray(
    llvm::LLVMContext & TheContext,
    llvm::Module &TheModule,
    const std::vector<T> & Vals,
    const std::vector<llvm::Constant*> & GEPIndices = {})
{
  using namespace llvm;
  std::vector<Constant*> ValsC;
  for (auto V : Vals)
    ValsC.emplace_back( llvmValue<U>(TheContext, V) );

  return llvmArray(TheContext, TheModule, ValsC, GEPIndices);
}

//============================================================================
void startLLVM();
void initializeAllTargets();
const llvm::Target* findTarget(const std::string&);

void insertModule(llvm::Module &, llvm::Module &);

} // namespace

#endif // CONFIG_HPP
