#ifndef LLVM_UTILS_HPP
#define LLVM_UTILS_HPP

#include "config.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"

#include <type_traits>

namespace {

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
auto llvmType( llvm::LLVMContext & TheContext )
{ return LlvmType<T>::getType(TheContext); }

template<typename T>
auto llvmValue( llvm::LLVMContext & TheContext, T Val )
{ return LlvmType<T>::getValue(TheContext, Val); }

inline auto llvmTypes(const std::vector<llvm::Value*> & Vals)
{
  std::vector<llvm::Type*> Types;
  Types.reserve(Vals.size());
  for (const auto & V : Vals) Types.emplace_back(V->getType());
  return Types;
}

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


//============================================================================  
inline llvm::IRBuilder<> createBuilder(llvm::Function *TheFunction)
{
  auto & Block = TheFunction->getEntryBlock();
  return llvm::IRBuilder<>(&Block, Block.begin());
}

//============================================================================  
inline llvm::AllocaInst* createEntryBlockAlloca(llvm::Function *TheFunction,
  llvm::Type* Ty, const std::string & Name)
{
  auto TmpB = createBuilder(TheFunction);
  return TmpB.CreateAlloca(Ty, nullptr, Name.c_str());
}

//============================================================================  
template<typename T>
llvm::Value* getTypeSize(llvm::IRBuilder<> & Builder, llvm::Type* ElementType)
{
  using namespace llvm;
  auto & TheContext = Builder.getContext();
  auto TheBlock = Builder.GetInsertBlock();
  auto PtrType = ElementType->getPointerTo();
  auto Index = ConstantInt::get(TheContext, APInt(32, 1, true));
  auto Null = Constant::getNullValue(PtrType);
  auto SizeGEP = Builder.CreateGEP(ElementType, Null, Index, "size");
  auto DataSize = CastInst::Create(Instruction::PtrToInt, SizeGEP,
          llvmType<T>(TheContext), "sizei", TheBlock);
  return DataSize;
}

} // namespace

#endif // CONFIG_HPP
