#include "serializer.hpp"

#include "contra/errors.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Support/raw_ostream.h"

namespace contra {

using namespace utils;
using namespace llvm;

//==============================================================================
// Constructor
//==============================================================================
Serializer::Serializer(BuilderHelper & TheHelper) :
  TheHelper_(TheHelper),
  Builder_(TheHelper.getBuilder()),
  TheContext_(TheHelper.getContext()),
  SizeType_(llvmType<size_t>(TheContext_))
{}

//==============================================================================
// Offset a pointer
//==============================================================================
Value* Serializer::offsetPointer(Value* Ptr, Value* Offset)
{
  Value* OffsetV = Offset;
  if (Offset->getType()->isPointerTy()) {
    THROW_CONTRA_ERROR("getPointerElementType");
    auto OffsetT = Offset->getType();//->getPointerElementType();
    OffsetV = Builder_.CreateLoad(OffsetT, Offset);
  }
  auto PtrV = TheHelper_.getAsValue(Ptr);
  THROW_CONTRA_ERROR("CreateGEP");
  return Builder_.CreateGEP(Ptr->getType(), PtrV, OffsetV);
}

//==============================================================================
// Default serializer
//==============================================================================
Value* Serializer::getSize(llvm::Module&, Value* Val, Type* ResultT)
{ return TheHelper_.getTypeSize(Val, ResultT); }

Value* Serializer::serialize(
    llvm::Module& TheModule,
    Value* SrcA,
    Value* TgtPtrV,
    Type* TgtT,
    Value* OffsetA)
{
  auto OffsetTgtPtrV = TgtPtrV;
  if (OffsetA) OffsetTgtPtrV = TheHelper_.offsetPointer(TgtT, TgtPtrV, OffsetA);
  auto SizeV = getSize(TheModule, SrcA, SizeType_);
  TheHelper_.memCopy(OffsetTgtPtrV, SrcA, SizeV); 
  return SizeV;
}

Value* Serializer::deserialize(
    llvm::Module& TheModule,
    AllocaInst* TgtA,
    Value* SrcA,
    Type* SrcT,
    Value* OffsetA)
{
  auto OffsetSrc = SrcA;
  if (OffsetA) OffsetSrc = TheHelper_.offsetPointer(SrcT, SrcA, OffsetA);
  auto SizeV = getSize(TheModule, TgtA, SizeType_);
  TheHelper_.memCopy(TgtA, OffsetSrc, SizeV);
  return SizeV;
}

//==============================================================================
// Array serializer
//==============================================================================
  
ArraySerializer::ArraySerializer(
    BuilderHelper & TheHelper,
    StructType* ArrayType) :
  Serializer(TheHelper),
  PtrType_(ArrayType->getElementType(0)),
  LengthType_(ArrayType->getElementType(1))
{}

Value* ArraySerializer::getSize(llvm::Module&, Value* Val, Type* ResultT)
{
  auto SizeV = TheHelper_.extractValue(Val, 1);
  auto DataSizeV = TheHelper_.extractValue(Val, 3);
  auto LenV = Builder_.CreateMul(SizeV, DataSizeV);
  auto IntSizeV = TheHelper_.getTypeSize(LengthType_, LengthType_);
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  if (LenV->getType() != ResultT) 
    LenV = TheHelper_.createCast(LenV, ResultT);
  return LenV;
}

Value* ArraySerializer::serialize(
    llvm::Module& TheModule,
    Value* SrcA,
    Value* TgtPtrV,
    Type* TgtT,
    Value* OffsetA)
{
  auto OffsetTgtPtrV = TgtPtrV;
  if (OffsetA) OffsetTgtPtrV = TheHelper_.offsetPointer(TgtT, TgtPtrV, OffsetA);
  // store size 
  auto LengthPtrT = LengthType_->getPointerTo();
  auto SizeV = TheHelper_.extractValue(SrcA, 1);
  auto SizeTgtPtr = TheHelper_.createBitCast(OffsetTgtPtrV, LengthPtrT);
  Builder_.CreateStore(SizeV, SizeTgtPtr);
  // increment
  auto IntSizeV = TheHelper_.getTypeSize(LengthType_, LengthType_);
  OffsetTgtPtrV = TheHelper_.offsetPointer(TgtT, OffsetTgtPtrV, IntSizeV);
  // store data size
  auto DataSizeV = TheHelper_.extractValue(SrcA, 3);
  auto DataSizeTgtPtr = TheHelper_.createBitCast(OffsetTgtPtrV, LengthPtrT);
  Builder_.CreateStore(DataSizeV, DataSizeTgtPtr);
  // increment
  OffsetTgtPtrV = TheHelper_.offsetPointer(TgtT, OffsetTgtPtrV, IntSizeV);
  // copy data
  auto LenV = Builder_.CreateMul(SizeV, DataSizeV);
  auto DataPtrV = TheHelper_.extractValue(SrcA, 0); 
  TheHelper_.memCopy(OffsetTgtPtrV, DataPtrV, LenV); 
  // return len
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  if (LenV->getType() != SizeType_) 
    LenV = TheHelper_.createCast(LenV, SizeType_);
  return LenV;
}

Value* ArraySerializer::deserialize(
    llvm::Module& TheModule,
    AllocaInst* TgtA,
    Value* SrcA,
    Type* SrcT,
    Value* OffsetA)
{
  // offset source pointer
  auto OffsetSrc = SrcA;
  if (OffsetA) OffsetSrc = TheHelper_.offsetPointer(SrcT, SrcA, OffsetA);
  // get size
  auto LengthPtrT = LengthType_->getPointerTo();
  auto SizePtrV = TheHelper_.createBitCast(OffsetSrc, LengthPtrT);
  Value* SizeV = Builder_.CreateLoad(LengthType_, SizePtrV);
  // increment
  auto IntSizeV = TheHelper_.getTypeSize(LengthType_, LengthType_);
  OffsetSrc = TheHelper_.offsetPointer(SrcT, OffsetSrc, IntSizeV);
  // get data size
  auto DataSizePtrV = TheHelper_.createBitCast(OffsetSrc, LengthPtrT);
  Value* DataSizeV = Builder_.CreateLoad(LengthType_, DataSizePtrV);
  // increment
  OffsetSrc = TheHelper_.offsetPointer(SrcT, OffsetSrc, IntSizeV);
  // create array
  auto VoidT = Type::getVoidTy(TheContext_);
  TheHelper_.callFunction(
      TheModule,
      "dopevector_allocate",
      VoidT,
      {SizeV, DataSizeV, TgtA});
  // copy data
  auto LenV = Builder_.CreateMul(SizeV, DataSizeV);
  auto DataPtrV = TheHelper_.extractValue(TgtA, 0);
  TheHelper_.memCopy(DataPtrV, OffsetSrc, LenV);
  // return len
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  if (LenV->getType() != SizeType_) 
    LenV = TheHelper_.createCast(LenV, SizeType_);
  return LenV;
}

}
