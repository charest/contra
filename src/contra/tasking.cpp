#include "tasking.hpp"

#include "utils/llvm_utils.hpp"

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
Type* AbstractTasker::reduceStruct(
    StructType * StructT,
    const Module &TheModule) const
{
  auto NumElem = StructT->getNumElements();
  auto ElementTs = StructT->elements();
  if (NumElem == 1) return ElementTs[0];
  auto DL = std::make_unique<DataLayout>(&TheModule);
  auto BitWidth = DL->getTypeAllocSizeInBits(StructT);
  return IntegerType::get(TheContext_, BitWidth);
}

//==============================================================================
Type* AbstractTasker::reduceArray(
    ArrayType * ArrayT,
    const Module &TheModule) const
{
  auto NumElem = ArrayT->getNumElements();
  auto ElementT = ArrayT->getElementType();
  if (NumElem == 1) return ElementT;
  auto DL = std::make_unique<DataLayout>(&TheModule);
  auto BitWidth = DL->getTypeAllocSizeInBits(ArrayT);
  return IntegerType::get(TheContext_, BitWidth);
}

//==============================================================================
Value* AbstractTasker::sanitize(Value* V, const Module &TheModule) const
{
  auto T = V->getType();
  if (auto StrucT = dyn_cast<StructType>(T)) {
    auto TheBlock = Builder_.GetInsertBlock();
    auto NewT = reduceStruct(StrucT, TheModule);
    std::string Str = StrucT->hasName() ? StrucT->getName().str()+".cast" : "casttmp";
    auto Cast = CastInst::Create(CastInst::BitCast, V, NewT, Str, TheBlock);
    return Cast;
  }
  else {
    return V;
  }
}

//==============================================================================
void AbstractTasker::sanitize(
    std::vector<Value*> & Vs,
    const Module &TheModule ) const
{
  for (auto & V : Vs ) V = sanitize(V, TheModule);
}

//==============================================================================
Value* AbstractTasker::load(
    Value * Alloca,
    const Module &TheModule,
    std::string Str) const
{
  if (!Str.empty()) Str += ".";
  auto AllocaT = Alloca->getType();
  auto BaseT = AllocaT->getPointerElementType();
  if (auto StructT = dyn_cast<StructType>(BaseT)) {
    auto TheBlock = Builder_.GetInsertBlock();
    auto ReducedT = reduceStruct(StructT, TheModule);
    auto Cast = CastInst::Create(CastInst::BitCast, Alloca,
      ReducedT->getPointerTo(), Str+"alloca.cast", TheBlock);
    return Builder_.CreateLoad(ReducedT, Cast, Str);
  }
  else if (auto ArrayT = dyn_cast<ArrayType>(BaseT)) {
    auto TheBlock = Builder_.GetInsertBlock();
    auto ReducedT = reduceArray(ArrayT, TheModule);
    auto Cast = CastInst::Create(CastInst::BitCast, Alloca,
      ReducedT->getPointerTo(), Str+"alloca.cast", TheBlock);
    return Builder_.CreateLoad(ReducedT, Cast, Str);
  }
  else {
    return Builder_.CreateLoad(BaseT, Alloca, Str);
  }
}

//==============================================================================
void AbstractTasker::store(Value* Val, Value * Alloca) const
{
  auto BaseT = Alloca->getType()->getPointerElementType();
  if (isa<StructType>(BaseT)) {
    std::vector<Value*> MemberIndices(2);
    MemberIndices[0] = ConstantInt::get(TheContext_, APInt(32, 0, true));
    MemberIndices[1] = ConstantInt::get(TheContext_, APInt(32, 0, true));
    auto GEP = Builder_.CreateGEP( BaseT, Alloca, MemberIndices );
    Builder_.CreateStore(Val, GEP);
  }
  else {
    Builder_.CreateStore(Val, Alloca);
  }
}
//==============================================================================
Value* AbstractTasker::offsetPointer(
    AllocaInst* PointerA,
    AllocaInst* OffsetA,
    const std::string & Name)
{
  return utils::offsetPointer(Builder_, PointerA, OffsetA, Name);
}
  
//==============================================================================
void AbstractTasker::increment(
    Value* OffsetA,
    Value* IncrV,
    const std::string & Name)
{
  return utils::increment(Builder_, OffsetA, IncrV, Name);
}
 
//==============================================================================
void AbstractTasker::memCopy(
    Value* SrcGEP,
    AllocaInst* TgtA,
    Value* SizeV, 
    const std::string & Name)
{
  std::string Str = Name.empty() ? "" : Name + ".";
  auto TgtPtrT = TgtA->getType();
  auto TheBlock = Builder_.GetInsertBlock();
  auto SrcPtrC = CastInst::Create(CastInst::BitCast, SrcGEP, TgtPtrT, "casttmp", TheBlock);
  Builder_.CreateMemCpy(TgtA, 1, SrcPtrC, 1, SizeV); 
}

//==============================================================================
Value* AbstractTasker::accessStructMember(
    AllocaInst* StructA,
    int i,
    const std::string & Name)
{
  std::vector<Value*> MemberIndices = {
     ConstantInt::get(TheContext_, APInt(32, 0, true)),
     ConstantInt::get(TheContext_, APInt(32, i, true))
  };
  auto StructT = StructA->getAllocatedType();
  return Builder_.CreateGEP(StructT, StructA, MemberIndices, Name);
}

//==============================================================================
Value* AbstractTasker::loadStructMember(
    AllocaInst* StructA,
    int i,
    const std::string & Name)
{
  auto ValueGEP = accessStructMember(StructA, i, Name);
  auto ValueT = ValueGEP->getType()->getPointerElementType();
  return Builder_.CreateLoad(ValueT, ValueGEP, Name);
}

//==============================================================================
void AbstractTasker::storeStructMember(
    Value* ValueV,
    AllocaInst* StructA,
    int i,
    const std::string & Name)
{
  auto ValueGEP = accessStructMember(StructA, i, Name);
  Builder_.CreateStore(ValueV, ValueGEP );
}
  
//==============================================================================
Value* AbstractTasker::start(Module & TheModule, int Argc, char ** Argv)
{ 
  setStarted();
  return startRuntime(TheModule, Argc, Argv);
}

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name)
{
  auto Id = getNextId();
  auto it = TaskTable_.emplace(Name, TaskInfo(Id));
  return it.first->second;
}

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name, Function* F)
{
  auto TaskName = F->getName();
  auto Id = getNextId();
  auto it = TaskTable_.emplace(Name, TaskInfo(Id, TaskName, F));
  return it.first->second;
}
  
//==============================================================================
TaskInfo AbstractTasker::popTask(const std::string & Name)
{
  auto it = TaskTable_.find(Name);
  auto res = it->second;
  TaskTable_.erase(it);
  return res;
}

//==============================================================================
void AbstractTasker::destroyFutures(
    Module & TheModule,
    const std::vector<Value*> & Futures)
{
  for (auto Future : Futures )
    destroyFuture(TheModule, Future);
}

//==============================================================================
void AbstractTasker::destroyFields(
    Module & TheModule,
    const std::vector<Value*> & Fields)
{
  for (auto Field : Fields )
    destroyField(TheModule, Field);
}

//==============================================================================
void AbstractTasker::destroyRanges(
    Module & TheModule,
    const std::vector<Value*> & Ranges)
{
  for (auto IS : Ranges )
    destroyRange(TheModule, IS);
}

//==============================================================================
void AbstractTasker::destroyAccessors(
    Module & TheModule,
    const std::vector<Value*> & Accessors)
{
  for (auto Acc : Accessors )
    destroyAccessor(TheModule, Acc);
}

//==============================================================================
void AbstractTasker::preregisterTasks(Module & TheModule)
{
  for (const auto & task_pair : TaskTable_ )
    preregisterTask(TheModule, task_pair.first, task_pair.second);
}

//==============================================================================
void AbstractTasker::postregisterTasks(Module & TheModule)
{
  for (const auto & task_pair : TaskTable_ )
    postregisterTask(TheModule, task_pair.first, task_pair.second);
}
 
//==============================================================================
Value* AbstractTasker::getSize(Value* Val, Type* ResultT)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end()) return it->second->getSize(Val, ResultT);

  // default behaviour
  return DefaultSerializer_.getSize(Val, ResultT);
}

Value* AbstractTasker::serialize(Value* Val, Value* DataPtrV, Value* OffsetA)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->serialize(Val, DataPtrV, OffsetA);
  else
    return DefaultSerializer_.serialize(Val, DataPtrV, OffsetA);
}

Value* AbstractTasker::deserialize(
    AllocaInst* ValA,
    Value* DataPtrV,
    Value* OffsetA)
{
  auto ValT = ValA->getAllocatedType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->deserialize(ValA, DataPtrV, OffsetA);
  else
    return DefaultSerializer_.deserialize(ValA, DataPtrV, OffsetA);
}

} // namespace
