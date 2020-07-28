#include "errors.hpp"
#include "tasking.hpp"

#include "utils/llvm_utils.hpp"
#include "llvm/Support/raw_ostream.h"

namespace contra {

using namespace llvm;
using namespace utils;
  
//==============================================================================
// Contructor
//==============================================================================
AbstractTasker::AbstractTasker(BuilderHelper & TheHelper) :
  TheHelper_(TheHelper),
  Builder_(TheHelper.getBuilder()),
  TheContext_(TheHelper.getContext()),
  DefaultSerializer_(TheHelper)
{
  VoidType_ = llvmType<void>(TheContext_);
  VoidPtrType_ = llvmType<void*>(TheContext_);
  ByteType_ = VoidPtrType_->getPointerElementType();
  BoolType_ = llvmType<bool>(TheContext_);
  Int32Type_ = llvmType<int>(TheContext_);
  IntType_ = llvmType<int_t>(TheContext_);
  SizeType_ = llvmType<std::size_t>(TheContext_);
  RealType_ = llvmType<real_t>(TheContext_);
  
  DefaultIndexSpaceType_ = createDefaultIndexSpaceType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * AbstractTasker::createDefaultIndexSpaceType()
{
  std::vector<Type*> members = { IntType_, IntType_, IntType_ };
  auto NewType = StructType::create( TheContext_, members, "contra_index_space_t" );
  return NewType;
}

//==============================================================================
// Default Preamble
//==============================================================================
AbstractTasker::PreambleResult AbstractTasker::taskPreamble(
    Module &TheModule,
    const std::string & Name,
    Function* TaskF)
{
  
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", TaskF);
  Builder_.SetInsertPoint(BB);

  std::vector<AllocaInst*> TaskArgAs;

  for (auto & Arg : TaskF->args()) {
    // get arg type
    auto ArgT = Arg.getType();
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = TheHelper_.createEntryBlockAlloca(TaskF, ArgT, ArgN);
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    TaskArgAs.emplace_back(Alloca);
  }
  
  return {TaskF, TaskArgAs, nullptr}; 
}

//==============================================================================
// Create the function wrapper
//==============================================================================
void AbstractTasker::taskPostamble(
    Module &TheModule,
    Value* ResultV,
    bool)
{

  //----------------------------------------------------------------------------
  // Have return value
  if (ResultV && !ResultV->getType()->isVoidTy()) {
    ResultV = TheHelper_.getAsValue(ResultV);
    Builder_.CreateRet(ResultV);
  }
  else {
    Builder_.CreateRetVoid();
  }

}
 
//==============================================================================
// Launch a task
//==============================================================================
Value* AbstractTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    const std::vector<Value*> & Args)
{
  std::vector<Value*> ArgVs;
  for (auto Arg : Args)
    ArgVs.emplace_back( TheHelper_.getAsValue(Arg) );

  auto ResultT = TaskI.getReturnType();
  auto Res = TheHelper_.callFunction(
      TheModule,
      TaskI.getName(),
      ResultT,
      ArgVs);
  return Res;
}

//==============================================================================
// Is this an range type
//==============================================================================
bool AbstractTasker::isRange(Type* RangeT) const
{
  return (RangeT == DefaultIndexSpaceType_);
}

bool AbstractTasker::isRange(Value* RangeA) const
{
  auto RangeT = RangeA->getType();
  if (isa<AllocaInst>(RangeA)) RangeT = RangeT->getPointerElementType();
  return isRange(RangeT);
}


//==============================================================================
// create a range
//==============================================================================
AllocaInst* AbstractTasker::createRange(
    Module & TheModule,
    const std::string & Name,
    Value* StartV,
    Value* EndV,
    Value* StepV)
{
  auto IndexSpaceA = TheHelper_.createEntryBlockAlloca(DefaultIndexSpaceType_, "index");

  StartV = TheHelper_.getAsValue(StartV);
  EndV = TheHelper_.getAsValue(EndV);
  if (StepV) StepV = TheHelper_.getAsValue(StepV);

  TheHelper_.insertValue(IndexSpaceA, StartV, 0);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  EndV = Builder_.CreateAdd(EndV, OneC);
  TheHelper_.insertValue(IndexSpaceA, EndV, 1);
  if (!StepV) StepV = OneC;
  TheHelper_.insertValue(IndexSpaceA, StepV, 2);
  
  return IndexSpaceA;

}

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* AbstractTasker::getRangeStart(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 0); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* AbstractTasker::getRangeEnd(Value* RangeV)
{
  Value* EndV = TheHelper_.extractValue(RangeV, 1);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  return Builder_.CreateSub(EndV, OneC);
}


//==============================================================================
// get a range start
//==============================================================================
llvm::Value* AbstractTasker::getRangeEndPlusOne(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 1); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* AbstractTasker::getRangeStep(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 2); }

//==============================================================================
// get a range size
//==============================================================================
llvm::Value* AbstractTasker::getRangeSize(Value* RangeV)
{
  auto StartV = TheHelper_.extractValue(RangeV, 0);
  auto EndV = TheHelper_.extractValue(RangeV, 1);
  return Builder_.CreateSub(EndV, StartV);
}

//==============================================================================
// get a range value
//==============================================================================
llvm::Value* AbstractTasker::loadRangeValue(
    Value* RangeA,
    Value* IndexV)
{
  auto StartV = TheHelper_.extractValue(RangeA, 0); 
  IndexV = TheHelper_.getAsValue(IndexV);
  return Builder_.CreateAdd(StartV, IndexV);
}

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
void AbstractTasker::start(Module & TheModule, int Argc, char ** Argv)
{ 
  setStarted();
  startRuntime(TheModule, Argc, Argv);
}

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name, Function* F)
{
  auto TaskName = F->getName().str();
  auto Id = makeTaskId();
  auto it = TaskTable_.emplace(Name, TaskInfo(Id, TaskName, F));
  return it.first->second;
}

//==============================================================================
const TaskInfo & AbstractTasker::getTask(const std::string & Name) const
{
  auto it = TaskTable_.find(Name);
  if (it != TaskTable_.end())
    return it->second;
  else 
    THROW_CONTRA_ERROR("Unknown task requested: '" << Name << "'.");
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

//==============================================================================
Constant* AbstractTasker::initReduce(Type* VarT, ReductionType Op)
{
  Constant* InitC = nullptr;
    
  constexpr auto MinReal = std::numeric_limits<real_t>::lowest();
  constexpr auto MaxReal = std::numeric_limits<real_t>::max();
  constexpr auto MinInt  = std::numeric_limits<int_t>::lowest();
  constexpr auto MaxInt  = std::numeric_limits<int_t>::max();

  // Floating point
  if (VarT->isFloatingPointTy()) {
    if (Op == ReductionType::Add ||
        Op == ReductionType::Sub)
      InitC = llvmValue<real_t>(TheContext_, 0);
    else if (Op == ReductionType::Mult ||
             Op == ReductionType::Div)
      InitC = llvmValue<real_t>(TheContext_, 1);
    else if (Op == ReductionType::Min)
      InitC = llvmValue<real_t>(TheContext_, MaxReal);
    else if (Op == ReductionType::Max)
      InitC = llvmValue<real_t>(TheContext_, MinReal);
    else {
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
  // Integer
  else {
    if (Op == ReductionType::Add ||
        Op == ReductionType::Sub)
      InitC = llvmValue<int_t>(TheContext_, 0);
    else if (Op == ReductionType::Mult ||
             Op == ReductionType::Div)
      InitC = llvmValue<int_t>(TheContext_, 1);
    else if (Op == ReductionType::Min)
      InitC = llvmValue<int_t>(TheContext_, MaxInt);
    else if (Op == ReductionType::Max)
      InitC = llvmValue<int_t>(TheContext_, MinInt);
    else {
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }

  return InitC;
}

//==============================================================================
Value* AbstractTasker::applyReduce(
    Module& TheModule,
    Value* LhsV,
    Value* RhsV,
    ReductionType Op)
{
  auto VarT = LhsV->getType();
  // Floating point
  if (VarT->isFloatingPointTy()) {
    switch (Op) {
    case ReductionType::Add:
      return Builder_.CreateFAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Sub:
      return Builder_.CreateFSub(LhsV, RhsV, "subtmp");
    case ReductionType::Mult:
      return Builder_.CreateFMul(LhsV, RhsV, "multmp");
    case ReductionType::Div:
      return Builder_.CreateFDiv(LhsV, RhsV, "divtmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(LhsV, RhsV, "max");
    default :
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
  // Integer
  else {
    switch (Op) {
    case ReductionType::Add:
      return Builder_.CreateAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Sub:
      return Builder_.CreateSub(LhsV, RhsV, "subtmp");
    case ReductionType::Mult:
      return Builder_.CreateMul(LhsV, RhsV, "multmp");
    case ReductionType::Div:
      return Builder_.CreateSDiv(LhsV, RhsV, "divtmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(LhsV, RhsV, "max");
    default:
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
    
  return nullptr;
}

//==============================================================================
Value* AbstractTasker::foldReduce(
    Module& TheModule,
    Value* LhsV,
    Value* RhsV,
    ReductionType Op)
{
  auto VarT = LhsV->getType();
  // Floating point
  if (VarT->isFloatingPointTy()) {
    switch (Op) {
    case ReductionType::Add:
    case ReductionType::Sub:
      return Builder_.CreateFAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Mult:
    case ReductionType::Div:
      return Builder_.CreateFMul(LhsV, RhsV, "multmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(LhsV, RhsV, "max");
    default :
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
  // Integer
  else {
    switch (Op) {
    case ReductionType::Add:
    case ReductionType::Sub:
      return Builder_.CreateAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Mult:
    case ReductionType::Div:
      return Builder_.CreateMul(LhsV, RhsV, "multmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(LhsV, RhsV, "max");
    default:
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
    
  return nullptr;
}

} // namespace
