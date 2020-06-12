#include "serializer.hpp"

#include "utils/llvm_utils.hpp"

namespace contra {

using namespace utils;
using namespace llvm;

Serializer::Serializer(BuilderHelper & TheHelper) :
  TheHelper_(TheHelper),
  Builder_(TheHelper.getBuilder()),
  TheContext_(TheHelper.getContext()),
  SizeType_(llvmType<size_t>(TheContext_))
{}

Value* Serializer::getSize(Value* Val, Type* ResultT)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  return TheHelper_.getTypeSize(ValT, ResultT );
}

Value* Serializer::offsetPointer(Value* Ptr, Value* Offset)
{
  Value* OffsetV = Offset;
  if (Offset->getType()->isPointerTy()) {
    auto OffsetT = Offset->getType()->getPointerElementType();
    OffsetV = Builder_.CreateLoad(OffsetT, Offset);
  }
  auto PtrV = TheHelper_.getAsValue(Ptr);
  return Builder_.CreateGEP(PtrV, OffsetV);
}

Value* Serializer::serialize(Value* SrcA, Value* TgtPtrV, Value* OffsetA)
{
  auto OffsetTgtPtrV = TgtPtrV;
  if (OffsetA) OffsetTgtPtrV = offsetPointer(TgtPtrV, OffsetA);
  auto SizeV = getSize(SrcA, SizeType_);
  Builder_.CreateMemCpy(OffsetTgtPtrV, 1, SrcA, 1, SizeV); 
  return SizeV;
}

Value* Serializer::deserialize(AllocaInst* TgtA, Value* SrcA, Value* OffsetA)
{
  auto OffsetSrc = SrcA;
  if (OffsetA) OffsetSrc = offsetPointer(SrcA, OffsetA);
  auto SizeV = getSize(TgtA, SizeType_);
  Builder_.CreateMemCpy(TgtA, 1, OffsetSrc, 1, SizeV);
  return SizeV;
}

}
