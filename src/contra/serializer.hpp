#ifndef CONTRA_SERIALIZER_HPP
#define CONTRA_SERIALIZER_HPP

#include "utils/builder.hpp"

#include <string>

namespace contra {

//==============================================================================
// Serializer
//==============================================================================
class Serializer {
  
  utils::Builder & TheBuilder_;
  
  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

  llvm::Type* SizeType_ = nullptr;

public:

  Serializer(utils::Builder & TheBuilder);

  llvm::Value* getSize(llvm::Value*, llvm::Type*);
  llvm::Value* offsetPointer(llvm::Value*, llvm::Value*);

  llvm::Value* serialize(llvm::Value*, llvm::Value*, llvm::Value*);
  llvm::Value* deserialize(llvm::AllocaInst*, llvm::Value*, llvm::Value*);

  virtual ~Serializer() = default;
};

#if 0
//==============================================================================
Value* AbstractTasker::getArraySize(Value* Val)
{
  if (isa<AllocaInst>(Val)) {
    auto ValT = Val->getType()->getPointerElementType();
    Val = Builder_.CreateLoad(ValT, Val);
  }
  auto SizeV = Builder_.CreateExtractValue(Val, 1);
  auto DataSizeV = Builder_.CreateExtractValue(Val, 3);
  auto LenV = Builder_.CreateMul(SizeV, DataSizeV);
  auto IntSizeV = getTypeSize<size_t>(Builder_, IntType_);
  auto VoidPtrSizeV = getTypeSize<size_t>(Builder_, VoidPtrType_);
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  LenV = Builder_.CreateAdd(LenV, IntSizeV);
  return LenV;
}

void AbstractTasker::serializeArray(Value*, Value*, Value*)
{
  auto SizeT = OffsetA->getType()->getPointerElementType();
  // size
  Value* OffsetV = Builder_.CreateLoad(SizeT, OffsetA);
  auto OffsetDataPtrV = Builder_.CreateGEP(DataPtrV, OffsetV);
  auto IntSizeV = getTypeSize(Builder_, SizeT, IntType_);
  auto SizeV = Builder_.CreateExtractValue(Val, 1);
  Builder_.CreateMemCpy(OffsetDataPtrV, 1, SizeV, 1, IntSizeV);
  // increment
  OffsetV = Builder_.CreateLoad(SizeT, OffsetA);
  OffsetV = Builder_.CreateAdd(OffsetV, IntSizeV);
  Builder_.CreateStore(OffsetV, OffsetA);
  // data_size
  OffsetV = Builder_.CreateLoad(SizeT, OffsetA);
  OffsetDataPtrV = Builder_.CreateGEP(DataPtrV, OffsetV);
  auto DataSizeV = Builder_.CreateExtractValue(Val, 3);
  Builder_.CreateMemCpy(OffsetDataPtrV, 1, DataSizeV, 1, IntSizeV);
  // increment
  OffsetV = Builder_.CreateLoad(SizeT, OffsetA);
  OffsetV = Builder_.CreateAdd(OffsetV, IntSizeV);
  Builder_.CreateStore(OffsetV, OffsetA);
  // data
  OffsetV = Builder_.CreateLoad(SizeT, OffsetA);
  OffsetDataPtrV = Builder_.CreateGEP(DataPtrV, OffsetV);
  auto LenV = Builder_.CreateMul(SizeV, DataSizeV);
  auto DataV = Builder_.CreateExtractValue(Val, 0);
  Builder_.CreateMemCpy(OffsetDataPtrV, 1, DataV, 1, LenV);
}

void AbstractTasker::deserializeArray(AllocaInst*, Value*, Value*)
{
  auto SizeT = OffsetA->getType()->getPointerElementType();
  auto TheBlock = Builder_.GetInsertBlock();
  // get size
  Value* OffsetV = Builder_.CreateLoad(SizeT, OffsetA);
  auto DataPtrV = Builder_.CreateLoad(PointerT, DataPtrA);
  auto OffsetGEP = Builder_.CreateGEP(DataPtrV, OffsetV);
  auto SizeA = createEntryBlockAlloca(TheFunction, IntType_);
  auto SrcPtrC = CastInst::Create(CastInst::BitCast, OffsetGEP, SizeA->getType(), "casttmp", TheBlock);
  Builder_.CreateMemCpy(TgtA, 1, SrcPtrC, 1, SizeV);
}

#endif

} // namespace

#endif // CONTRA_SERIALIZER_HPP
