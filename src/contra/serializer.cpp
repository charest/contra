#include "serializer.hpp"

#include "utils/llvm_utils.hpp"

namespace contra {

using namespace utils;
using namespace llvm;

Serializer::Serializer(
    llvm::IRBuilder<> & TheBuilder,
    llvm::LLVMContext & TheContext) :
  Builder_(TheBuilder),
  TheContext_(TheContext),
  SizeType_(llvmType<size_t>(TheContext_))
{}

Value* Serializer::getSize(Value* Val, Type* ResultT)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  return utils::getTypeSize(Builder_, ValT, ResultT );
}

Value* Serializer::serialize(Value* SrcA, Value* TgtPtrV, Value* OffsetA)
{
  auto OffsetTgtPtrV = TgtPtrV;
  if (OffsetA) OffsetTgtPtrV = offsetPointer(Builder_, TgtPtrV, OffsetA);
  auto SizeV = getSize(SrcA, SizeType_);
  Builder_.CreateMemCpy(OffsetTgtPtrV, 1, SrcA, 1, SizeV); 
  return SizeV;
}

Value* Serializer::deserialize(AllocaInst* TgtA, Value* SrcA, Value* OffsetA)
{
  auto OffsetSrc = SrcA;
  if (OffsetA) OffsetSrc = offsetPointer(Builder_, SrcA, OffsetA);
  auto SizeV = getSize(TgtA, SizeType_);
  Builder_.CreateMemCpy(TgtA, 1, OffsetSrc, 1, SizeV);
  return SizeV;
}

}
