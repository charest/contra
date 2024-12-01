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
  DefaultSerializer_(TheHelper)
{
  VoidType_ = llvmType<void>(getContext());
  VoidPtrType_ = llvmType<void*>(getContext());
  ByteType_ = llvmType<int8_t>(getContext());
  BoolType_ = llvmType<bool>(getContext());
  Int32Type_ = llvmType<int>(getContext());
  Int1Type_ = Type::getInt1Ty(getContext());
  IntType_ = llvmType<int_t>(getContext());
  SizeType_ = llvmType<std::size_t>(getContext());
  RealType_ = llvmType<real_t>(getContext());
  
  DefaultIndexSpaceType_ = createDefaultIndexSpaceType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * AbstractTasker::createDefaultIndexSpaceType()
{
  std::vector<Type*> members = { IntType_, IntType_, IntType_ };
  auto NewType = StructType::create( getContext(), members, "contra_index_space_t" );
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
  BasicBlock *BB = BasicBlock::Create(getContext(), "entry", TaskF);
  getBuilder().SetInsertPoint(BB);

  std::vector<AllocaInst*> TaskArgAs;

  for (auto & Arg : TaskF->args()) {
    // get arg type
    auto ArgT = Arg.getType();
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = TheHelper_.createEntryBlockAlloca(TaskF, ArgT, ArgN);
    // Store the initial value into the alloca.
    getBuilder().CreateStore(&Arg, Alloca);
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
    getBuilder().CreateRet(ResultV);
  }
  else {
    getBuilder().CreateRetVoid();
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
  if (isa<AllocaInst>(RangeA))
    RangeT = cast<AllocaInst>(RangeA)->getAllocatedType();
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
  auto OneC = llvmValue<int_t>(getContext(), 1);
  EndV = getBuilder().CreateAdd(EndV, OneC);
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
  auto OneC = llvmValue<int_t>(getContext(), 1);
  return getBuilder().CreateSub(EndV, OneC);
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
  return getBuilder().CreateSub(EndV, StartV);
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
  return getBuilder().CreateAdd(StartV, IndexV);
}

//==============================================================================
Type* AbstractTasker::reduceStruct(
    StructType * StructT,
    const Module &TheModule)
{
  auto NumElem = StructT->getNumElements();
  auto ElementTs = StructT->elements();
  if (NumElem == 1) return ElementTs[0];
  auto BitWidth = TheHelper_.getTypeSizeInBits(TheModule, StructT);
  return IntegerType::get(getContext(), BitWidth);
}

//==============================================================================
Type* AbstractTasker::reduceArray(
    ArrayType * ArrayT,
    const Module &TheModule)
{
  auto NumElem = ArrayT->getNumElements();
  auto ElementT = ArrayT->getElementType();
  if (NumElem == 1) return ElementT;
  auto BitWidth = TheHelper_.getTypeSizeInBits(TheModule, ArrayT);
  return IntegerType::get(getContext(), BitWidth);
}

//==============================================================================
Value* AbstractTasker::sanitize(Value* V, const Module &TheModule)
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
    const Module &TheModule )
{
  for (auto & V : Vs ) V = sanitize(V, TheModule);
}

//==============================================================================
Value* AbstractTasker::load(
    Type* BaseT,
    Value* Alloca,
    const Module &TheModule,
    std::string Str)
{
  if (!Str.empty()) Str += ".";
  if (auto StructT = dyn_cast<StructType>(BaseT)) {
    auto ReducedT = reduceStruct(StructT, TheModule);
    auto Cast = TheHelper_.createBitCast(Alloca, ReducedT->getPointerTo());
    return getBuilder().CreateLoad(ReducedT, Cast, Str);
  }
  else if (auto ArrayT = dyn_cast<ArrayType>(BaseT)) {
    auto ReducedT = reduceArray(ArrayT, TheModule);
    auto Cast = TheHelper_.createBitCast(Alloca, ReducedT->getPointerTo());
    return getBuilder().CreateLoad(ReducedT, Cast, Str);
  }
  else {
    return getBuilder().CreateLoad(BaseT, Alloca, Str);
  }
}

Value* AbstractTasker::load(
    AllocaInst* Alloca,
    const Module &TheModule,
    std::string Str)
{
  return load(Alloca->getAllocatedType(), Alloca, TheModule, Str);
}

//==============================================================================
void AbstractTasker::store(Type* BaseT, Value* Val, Value * Alloca)
{
  if (isa<StructType>(BaseT)) {
    std::vector<Value*> MemberIndices(2);
    MemberIndices[0] = ConstantInt::get(getContext(), APInt(32, 0, true));
    MemberIndices[1] = ConstantInt::get(getContext(), APInt(32, 0, true));
    auto GEP = getBuilder().CreateGEP( BaseT, Alloca, MemberIndices );
    getBuilder().CreateStore(Val, GEP);
  }
  else {
    getBuilder().CreateStore(Val, Alloca);
  }
}

void AbstractTasker::store(Value* Val, AllocaInst * Alloca)
{ store(Alloca->getAllocatedType(), Val, Alloca); }

//==============================================================================
void AbstractTasker::start(Module & TheModule)
{ 
  setStarted();
  startRuntime(TheModule);
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
  if (auto ValA = dyn_cast<AllocaInst>(Val))
    ValT = ValA->getAllocatedType();//->getPointerElementType();
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
    Type* DataT,
    Type* OffsetT,
    Value* OffsetA)
{
  auto ValT = Val->getType();
  if (auto ValA = dyn_cast<AllocaInst>(Val))
    ValT = ValA->getAllocatedType();//->getPointerElementType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->serialize(TheModule, Val, DataPtrV, DataT, OffsetT, OffsetA);
  else
    return DefaultSerializer_.serialize(TheModule, Val, DataPtrV, DataT, OffsetT, OffsetA);
}

Value* AbstractTasker::deserialize(
    Module& TheModule,
    AllocaInst* ValA,
    Value* DataPtrV,
    Type* DataT,
    Type* OffsetT,
    Value* OffsetA)
{
  auto ValT = ValA->getAllocatedType();
  auto it = Serializer_.find(ValT);
  if (it != Serializer_.end())
    return it->second->deserialize(TheModule, ValA, DataPtrV, DataT, OffsetT, OffsetA);
  else
    return DefaultSerializer_.deserialize(TheModule, ValA, DataPtrV, DataT, OffsetT, OffsetA);
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
      InitC = llvmValue<real_t>(getContext(), 0);
    else if (Op == ReductionType::Mult ||
             Op == ReductionType::Div)
      InitC = llvmValue<real_t>(getContext(), 1);
    else if (Op == ReductionType::Min)
      InitC = llvmValue<real_t>(getContext(), MaxReal);
    else if (Op == ReductionType::Max)
      InitC = llvmValue<real_t>(getContext(), MinReal);
    else {
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
  // Integer
  else {
    if (Op == ReductionType::Add ||
        Op == ReductionType::Sub)
      InitC = llvmValue<int_t>(getContext(), 0);
    else if (Op == ReductionType::Mult ||
             Op == ReductionType::Div)
      InitC = llvmValue<int_t>(getContext(), 1);
    else if (Op == ReductionType::Min)
      InitC = llvmValue<int_t>(getContext(), MaxInt);
    else if (Op == ReductionType::Max)
      InitC = llvmValue<int_t>(getContext(), MinInt);
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
      return getBuilder().CreateFAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Sub:
      return getBuilder().CreateFSub(LhsV, RhsV, "subtmp");
    case ReductionType::Mult:
      return getBuilder().CreateFMul(LhsV, RhsV, "multmp");
    case ReductionType::Div:
      return getBuilder().CreateFDiv(LhsV, RhsV, "divtmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(TheModule, LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(TheModule, LhsV, RhsV, "max");
    default :
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
  // Integer
  else {
    switch (Op) {
    case ReductionType::Add:
      return getBuilder().CreateAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Sub:
      return getBuilder().CreateSub(LhsV, RhsV, "subtmp");
    case ReductionType::Mult:
      return getBuilder().CreateMul(LhsV, RhsV, "multmp");
    case ReductionType::Div:
      return getBuilder().CreateSDiv(LhsV, RhsV, "divtmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(TheModule, LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(TheModule, LhsV, RhsV, "max");
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
      return getBuilder().CreateFAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Mult:
    case ReductionType::Div:
      return getBuilder().CreateFMul(LhsV, RhsV, "multmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(TheModule, LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(TheModule, LhsV, RhsV, "max");
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
      return getBuilder().CreateAdd(LhsV, RhsV, "addtmp");
    case ReductionType::Mult:
    case ReductionType::Div:
      return getBuilder().CreateMul(LhsV, RhsV, "multmp");
    case ReductionType::Min:
      return TheHelper_.createMinimum(TheModule, LhsV, RhsV, "min");
    case ReductionType::Max:
      return TheHelper_.createMaximum(TheModule, LhsV, RhsV, "max");
    default:
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
  }
    
  return nullptr;
}

} // namespace
