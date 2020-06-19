#include "builder.hpp"

#include "llvm_utils.hpp"

namespace utils {

using namespace llvm;


//==============================================================================
// Cast utility
//==============================================================================
Value* BuilderHelper::createCast(Value* FromVal, Type* ToType)
{
  auto FromType = FromVal->getType();
  auto TheBlock = Builder_.GetInsertBlock();

  if (FromType->isFloatingPointTy() && ToType->isIntegerTy()) {
    return CastInst::Create(Instruction::FPToSI, FromVal,
        ToType, "cast", TheBlock);
  }
  else if (FromType->isIntegerTy() && ToType->isFloatingPointTy()) {
    return CastInst::Create(Instruction::SIToFP, FromVal,
        ToType, "cast", TheBlock);
  }
  else {
    return FromVal;
  }
}

//==============================================================================
// Cast utility
//==============================================================================
Value* BuilderHelper::createBitCast(Value* FromVal, Type* ToType)
{
  auto TheBlock = Builder_.GetInsertBlock();
  return CastInst::Create(Instruction::BitCast, FromVal, ToType, "cast", TheBlock);
}


//==============================================================================
// Extract values from allocas
//==============================================================================
Value* BuilderHelper::getAsValue(Value* ValueV)
{
  if (auto ValueA = dyn_cast<AllocaInst>(ValueV)) {
    auto ValueT = ValueA->getAllocatedType();
    return Builder_.CreateLoad(ValueT, ValueA);
  }
  return ValueV;
}

Value* BuilderHelper::getAsValue(Value* ValueV, Type* ValueT)
{
  if (auto ValueA = dyn_cast<AllocaInst>(ValueV)) {
    return Builder_.CreateLoad(ValueT, ValueA);
  }
  return ValueV;
}


//============================================================================  
// copy value into alloca if necessary
//============================================================================  
AllocaInst* BuilderHelper::getAsAlloca(Value* ValueV)
{
  AllocaInst* ValueA = dyn_cast<AllocaInst>(ValueV);
  if (!ValueA) {
    auto ValueT = ValueV->getType();
    ValueA = createEntryBlockAlloca(ValueT);
    Builder_.CreateStore(ValueV, ValueA);
  }
  return ValueA;
}

//==============================================================================
// Get pointer to struct member
//==============================================================================
Value* BuilderHelper::getElementPointer(AllocaInst* ValA, unsigned i)
{
  std::vector<Value*> MemberIndices = {
    ConstantInt::get(TheContext_, APInt(32, 0, true)),
    ConstantInt::get(TheContext_, APInt(32, i, true))
  };
  auto ValT = ValA->getAllocatedType();
  return Builder_.CreateGEP(ValT, ValA, MemberIndices);
}
  
  
//==============================================================================
// Get pointer to struct member
//==============================================================================
Value* BuilderHelper::offsetPointer(Value* Ptr, Value* Offset)
{
  auto OffsetV = getAsValue(Offset);
  auto PtrV = getAsValue(Ptr);
  return Builder_.CreateGEP(PtrV, OffsetV);
}

//==============================================================================
// extract a struct value
//==============================================================================
Value* BuilderHelper::extractValue(Value* Val, unsigned i) {
  if (auto ValA = dyn_cast<AllocaInst>(Val)) {
    auto ValGEP = getElementPointer(ValA, i);
    auto MemberT = ValGEP->getType()->getPointerElementType();
    return Builder_.CreateLoad(MemberT, ValGEP);
  }
  else {
    return Builder_.CreateExtractValue(Val, i);
  }
}

//==============================================================================
// insert a struct value
//==============================================================================
void BuilderHelper::insertValue(Value* Val, Value* Member, unsigned i) {
  if (auto ValA = dyn_cast<AllocaInst>(Val)) {
    auto ValGEP = getElementPointer(ValA, i);
    auto MemberV = getAsValue(Member);
    Builder_.CreateStore(MemberV, ValGEP);
  }
  else {
    std::cerr << "You can't insert a value into a loaded struct." << std::endl;
    abort();
  }
}
  

//==============================================================================
// Get the base type of an alloca or value
//==============================================================================
Type* BuilderHelper::getAllocatedType(Value* Val)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  return ValT;
}

//==============================================================================
// get the allocated size of a type
//==============================================================================
Value* BuilderHelper::getTypeSize(Type* ValT, Type* ResultT)
{
  auto TheBlock = Builder_.GetInsertBlock();
  auto PtrT = ValT->getPointerTo();
  auto Index = ConstantInt::get(TheContext_, APInt(32, 1, true));
  auto Null = Constant::getNullValue(PtrT);
  auto SizeGEP = Builder_.CreateGEP(ValT, Null, Index, "size");
  auto DataSize = CastInst::Create(
      Instruction::PtrToInt,
      SizeGEP,
      ResultT,
      "sizei",
      TheBlock);
  return DataSize;
}

std::size_t BuilderHelper::getTypeSizeInBits(const Module & TheModule, Type* Ty)
{
  auto DL = std::make_unique<DataLayout>(&TheModule);
  return DL->getTypeAllocSizeInBits(Ty);
}

//============================================================================  
// Create an alloca
//============================================================================  
AllocaInst* BuilderHelper::createEntryBlockAlloca(
    Type* Ty,
    const std::string & Name)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  return createEntryBlockAlloca(TheFunction, Ty, Name);
}

AllocaInst* BuilderHelper::createEntryBlockAlloca(
    Function* F,
    Type* Ty,
    const std::string & Name)
{
  auto & Block = F->getEntryBlock();
  auto TmpB = IRBuilder<>(&Block, Block.begin());
  return TmpB.CreateAlloca(Ty, nullptr, Name.c_str());
}

//============================================================================  
// load an alloca
//============================================================================  
Value* BuilderHelper::load(
    AllocaInst* ValA,
    const std::string & Name)
{
  auto ValT = ValA->getAllocatedType();
  return Builder_.CreateLoad(ValT, ValA, Name);
}

Value* BuilderHelper::load(
    Value* ValA,
    const std::string & Name)
{
  auto ValT = ValA->getType()->getPointerElementType();
  return Builder_.CreateLoad(ValT, ValA, Name);
}


//============================================================================  
// increment a counter
//============================================================================  
void BuilderHelper::increment(
    Value* OffsetA,
    Value* Incr,
    const std::string & Name)
{
  std::string Str = Name.empty() ? "" : Name + ".";
  auto OffsetV = load(OffsetA);
  auto IncrV = getAsValue(Incr);
  auto NewOffsetV = Builder_.CreateAdd(OffsetV, IncrV, Str+"add");
  Builder_.CreateStore( NewOffsetV, OffsetA );
}

//============================================================================  
// Malloc
//============================================================================  
Instruction* BuilderHelper::createMalloc(
    Type * Ty,
    Value* Size,
    const std::string & Name)
{
  auto SizeV = getAsValue(Size);
  auto SizeT = SizeV->getType();

  // not needed but InsertAtEnd doesnt work
  auto TmpA = Builder_.CreateAlloca(Ty, nullptr);
  auto MallocI = CallInst::CreateMalloc(
      TmpA,
      SizeT,
      Ty,
      SizeV,
      nullptr,
      nullptr,
      Name );
  TmpA->eraseFromParent();
  return MallocI;
}

//============================================================================  
// Free
//============================================================================  
void BuilderHelper::createFree(Value* Val)
{
  auto Ty = Val->getType();

  // not needed but InsertAtEnd doesnt work
  auto TmpA = Builder_.CreateAlloca(Ty, nullptr);
  CallInst::CreateFree(Val, TmpA);
  TmpA->eraseFromParent();
}
  
//============================================================================  
// Create a function
//============================================================================  
FunctionCallee BuilderHelper::createFunction(
    Module & TheModule,
    const std::string & Name,
    Type* ReturnT,
    const std::vector<Type*> & ArgTypes)
{
  if (ArgTypes.empty()) {
    auto FunT = FunctionType::get(ReturnT, None, false);
    auto FunF = TheModule.getOrInsertFunction(Name, FunT);
    return FunF;
  }
  else {
    auto FunT = FunctionType::get(ReturnT, ArgTypes, false);
    auto FunF = TheModule.getOrInsertFunction(Name, FunT);
    return FunF;
  }
}

//============================================================================  
// Call a function
//============================================================================  
CallInst* BuilderHelper::callFunction(
    Module & TheModule,
    const std::string & Name,
    Type* ReturnT,
    const std::vector<Value*> &ArgVs,
    const std::string & Str)
{
  std::vector<Type*> ArgTs;
  ArgTs.reserve(ArgVs.size());
  for (auto Arg : ArgVs) ArgTs.emplace_back( Arg->getType() );
  auto FunF = createFunction(TheModule, Name, ReturnT, ArgTs);

  if (ArgVs.empty()) {
    if (ReturnT->isVoidTy())
      return Builder_.CreateCall(FunF);
    else
      return Builder_.CreateCall(FunF, None, Str);
  }
  else {
    if (ReturnT->isVoidTy())
      return Builder_.CreateCall(FunF, ArgVs);
    else
      return Builder_.CreateCall(FunF, ArgVs, Str);
  }
}

//==============================================================================
// Memcopy utility
//==============================================================================
CallInst* BuilderHelper::memCopy(
    Value* Dest,
    Value* Src,
    Value* Size)
{
  return Builder_.CreateMemCpy(Dest, MaybeAlign(1), Src, MaybeAlign(1), Size);
}

//==============================================================================
// Memset utility
//==============================================================================
CallInst* BuilderHelper::memSet(
    Value* Dest,
    Value* Src,
    unsigned Size)
{
  return Builder_.CreateMemSet(Dest, Src, Size, MaybeAlign(1));
}

} // namespace
