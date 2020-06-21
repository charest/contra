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
  auto BitWidth = TheHelper_.getTypeSizeInBits(TheModule, StructT);
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
  auto BitWidth = TheHelper_.getTypeSizeInBits(TheModule, ArrayT);
  return IntegerType::get(TheContext_, BitWidth);
}

//==============================================================================
Value* AbstractTasker::sanitize(Value* V, const Module &TheModule) const
{
  auto T = V->getType();
  if (auto StrucT = dyn_cast<StructType>(T)) {
    auto NewT = reduceStruct(StrucT, TheModule);
    return TheHelper_.createBitCast(V, NewT);
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
    auto ReducedT = reduceStruct(StructT, TheModule);
    auto Cast = TheHelper_.createBitCast(Alloca, ReducedT->getPointerTo());
    return Builder_.CreateLoad(ReducedT, Cast, Str);
  }
  else if (auto ArrayT = dyn_cast<ArrayType>(BaseT)) {
    auto ReducedT = reduceArray(ArrayT, TheModule);
    auto Cast = TheHelper_.createBitCast(Alloca, ReducedT->getPointerTo());
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
Value* AbstractTasker::start(Module & TheModule, int Argc, char ** Argv)
{ 
  setStarted();
  return startRuntime(TheModule, Argc, Argv);
}

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name)
{
  auto Id = makeTaskId();
  auto it = TaskTable_.emplace(Name, TaskInfo(Id));
  return it.first->second;
}

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name, Function* F)
{
  auto TaskName = F->getName();
  auto Id = makeTaskId();
  auto it = TaskTable_.emplace(Name, TaskInfo(Id, TaskName, F));
  return it.first->second;
}
  
//==============================================================================
TaskInfo AbstractTasker::popTask(const std::string & Name)
{
  auto it = TaskTable_.find(Name);
  auto res = std::move(it->second);
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
void AbstractTasker::destroyPartitions(
    Module & TheModule,
    const std::vector<Value*> & Parts)
{
  for (auto Part : Parts )
    destroyPartition(TheModule, Part);
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
Value* AbstractTasker::getSerializedSize(
    Module& TheModule,
    Value* Val,
    Type* ResultT)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->getSize(TheModule, Val, ResultT);

  // default behaviour
  return DefaultSerializer_.getSize(TheModule, Val, ResultT);
}

Value* AbstractTasker::serialize(
    Module& TheModule,
    Value* Val,
    Value* DataPtrV,
    Value* OffsetA)
{
  auto ValT = Val->getType();
  if (isa<AllocaInst>(Val)) ValT = Val->getType()->getPointerElementType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->serialize(TheModule, Val, DataPtrV, OffsetA);
  else
    return DefaultSerializer_.serialize(TheModule, Val, DataPtrV, OffsetA);
}

Value* AbstractTasker::deserialize(
    Module& TheModule,
    AllocaInst* ValA,
    Value* DataPtrV,
    Value* OffsetA)
{
  auto ValT = ValA->getAllocatedType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->deserialize(TheModule, ValA, DataPtrV, OffsetA);
  else
    return DefaultSerializer_.deserialize(TheModule, ValA, DataPtrV, OffsetA);
}

} // namespace
