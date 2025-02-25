#include "serial.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Support/raw_ostream.h"
  
////////////////////////////////////////////////////////////////////////////////
// Serial tasker
////////////////////////////////////////////////////////////////////////////////

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
// Constructor
//==============================================================================
SerialTasker::SerialTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  IndexSpaceType_ = DefaultIndexSpaceType_;
  IndexPartitionType_ = createIndexPartitionType();
  PartitionInfoType_ = VoidPtrType_->getPointerTo();
  FieldType_ = createFieldType();
  AccessorType_ = createAccessorType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * SerialTasker::createFieldType()
{
  std::vector<Type*> members = {
    IntType_,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo() };
  auto NewType = StructType::create( getContext(), members, "contra_serial_field_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * SerialTasker::createAccessorType()
{
  std::vector<Type*> members = {
    BoolType_,
    IntType_,
    VoidPtrType_};
  auto NewType = StructType::create( getContext(), members, "contra_serial_accessor_t" );
  return NewType;
}


//==============================================================================
// Create the partition data type
//==============================================================================
StructType * SerialTasker::createIndexPartitionType()
{
  auto IntPtrT = IntType_->getPointerTo();
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    IntPtrT,
    IntPtrT,
    IndexSpaceType_->getPointerTo()};
  auto NewType = StructType::create( getContext(), members, "contra_serial_partition_t" );
  return NewType;
}

//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* SerialTasker::createPartitionInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(PartitionInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_partition_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void SerialTasker::destroyPartitionInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_partition_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// start runtime
//==============================================================================
void SerialTasker::startRuntime(Module &TheModule)
{
  launch(TheModule, *TopLevelTask_);
}

//==============================================================================
// Create the function wrapper
//==============================================================================
SerialTasker::PreambleResult SerialTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    llvm::Type* ResultT)
{

  std::vector<std::string> WrapperArgNs = TaskArgNs;
    
  std::vector<Type*> WrapperArgTs;
  for (auto ArgT : TaskArgTs) {
    if (isRange(ArgT)) ArgT = IndexPartitionType_;
    WrapperArgTs.emplace_back(ArgT);
  }

  WrapperArgTs.emplace_back(IntType_);
  WrapperArgNs.emplace_back("index");

  if (!ResultT) ResultT = VoidType_;

  auto WrapperT = FunctionType::get(ResultT, WrapperArgTs, false);
  auto WrapperF = Function::Create(
      WrapperT,
      Function::ExternalLinkage,
      TaskName,
      &TheModule);
  
  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : WrapperF->args()) Arg.setName(WrapperArgNs[Idx++]);
  
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(getContext(), "entry", WrapperF);
  getBuilder().SetInsertPoint(BB);
  
  // allocate arguments
  std::vector<AllocaInst*> WrapperArgAs;
  WrapperArgAs.reserve(WrapperArgTs.size());

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    // Store the initial value into the alloca.
    getBuilder().CreateStore(&Arg, Alloca);
    WrapperArgAs.emplace_back(Alloca);
    ArgIdx++;
  }

  //----------------------------------------------------------------------------
  // partition any ranges
  std::vector<Type*> GetRangeArgTs = {
    IntType_,
    IndexPartitionType_->getPointerTo(),
    IndexSpaceType_->getPointerTo()
  };

  auto GetRangeF = TheHelper_.createFunction(
      TheModule,
      "contra_serial_index_space_create_from_partition",
      VoidType_,
      GetRangeArgTs);
  
  auto IndexA = WrapperArgAs.back();
  auto IndexT = IndexA->getAllocatedType();
  auto IndexV = getBuilder().CreateLoad(IndexT, IndexA);

  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    if (isRange(TaskArgTs[i])) {
      auto ArgN = TaskArgNs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      getBuilder().CreateCall(GetRangeF, {IndexV, WrapperArgAs[i], ArgA});
      WrapperArgAs[i] = ArgA;
    }
  }
  

  WrapperArgAs.pop_back();
  return {WrapperF, WrapperArgAs, IndexA};
}


//==============================================================================
// Launch an index task
//==============================================================================
Value* SerialTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    std::vector<Value*> ArgAs,
    const std::vector<Value*> & PartAs,
    Value* RangeV,
    const AbstractReduceInfo* AbstractReduceOp)
{
  auto PartInfoA = createPartitionInfo(TheModule);

  //----------------------------------------------------------------------------
  // Swap ranges for partitions

  std::vector<Value*> TempParts;

  auto NumArgs = ArgAs.size();
  for (unsigned i=0; i<NumArgs; i++) {
    if (isRange(ArgAs[i])) {
      // keep track of range
      auto IndexSpaceA = ArgAs[i];
      // has a prescribed partition
      if (PartAs[i]) {
        ArgAs[i] = PartAs[i];
      }
      // temporarily partition
      else {
        ArgAs[i] = createPartition(TheModule, ArgAs[i], RangeV);
        TempParts.emplace_back( ArgAs[i] );
      }
      // keep track of partition
      auto IndexPartitionA = ArgAs[i];
      // register these partitions
      std::vector<Value*> FunArgVs = {
        IndexSpaceA,
        IndexPartitionA,
        PartInfoA};
      TheHelper_.callFunction(
          TheModule,
          "contra_serial_register_index_partition",
          VoidType_,
          FunArgVs);
    }
  }

  //----------------------------------------------------------------------------
  // Prepare other args

  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  std::map<unsigned, std::pair<AllocaInst*, Value*>> AccessorData; 

  for (unsigned i=0; i<NumArgs; i++) {
    if (!isField(ArgAs[i])) continue;
    auto FieldA = TheHelper_.getAsAlloca(ArgAs[i]);
    Value* IndexPartitionA = nullptr;
    if (PartAs[i]) {
      IndexPartitionA = TheHelper_.getAsAlloca(PartAs[i]);
    }
    else {
      IndexPartitionA = TheHelper_.callFunction(
          TheModule,
          "contra_serial_partition_get",
          IndexPartitionType_->getPointerTo(),
          {IndexSpaceA, FieldA, PartInfoA});
    }
    auto AccessorA = TheHelper_.createEntryBlockAlloca(AccessorType_);
    AccessorData.emplace( i, std::make_pair(AccessorA, IndexPartitionA) );
  }
  
  //----------------------------------------------------------------------------
  // Do reductions
  AllocaInst* ResultA = nullptr;
  
  if (AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const SerialReduceInfo*>(AbstractReduceOp);
    
    auto ResultT = StructType::create( getContext(), "reduce_t" );
    ResultT->setBody( ReduceOp->getVarTypes() );
    ResultA = TheHelper_.createEntryBlockAlloca(ResultT);

    auto NumReduce = ReduceOp->getNumReductions();
    for (unsigned i=0; i<NumReduce; ++i) {
      auto VarT = ReduceOp->getVarType(i);
      auto Op = ReduceOp->getReduceOp(i);
      // get init value
      auto InitC = initReduce(VarT, Op);
      // store init
      TheHelper_.insertValue(ResultA, InitC, i);
    }

  }
  
  //----------------------------------------------------------------------------
  // create for loop
  
  // Create an alloca for the variable in the entry block.
  auto VarT = IntType_;
  auto VarA = TheHelper_.createEntryBlockAlloca(VarT, "index");
  
  // Emit the start code first, without 'variable' in scope.
  auto EndA = TheHelper_.createEntryBlockAlloca(VarT, "end");
  auto StepA = TheHelper_.createEntryBlockAlloca(VarT, "step");

  auto StartV = getRangeStart(RangeV);
  getBuilder().CreateStore(StartV, VarA);
  auto EndV = getRangeEndPlusOne(RangeV);
  getBuilder().CreateStore(EndV, EndA);
  auto StepV = getRangeStep(RangeV);
  getBuilder().CreateStore(StepV, StepA);
  
  // Make the new basic block for the loop header, inserting after current
  // block.
  auto TheFunction = getBuilder().GetInsertBlock()->getParent();
  BasicBlock *BeforeBB = BasicBlock::Create(getContext(), "beforeloop", TheFunction);
  BasicBlock *LoopBB =   BasicBlock::Create(getContext(), "loop", TheFunction);
  BasicBlock *IncrBB =   BasicBlock::Create(getContext(), "incr", TheFunction);
  BasicBlock *AfterBB =  BasicBlock::Create(getContext(), "afterloop", TheFunction);
  
  getBuilder().CreateBr(BeforeBB);
  getBuilder().SetInsertPoint(BeforeBB);

  // Load value and check coondition
  Value *CurV = getBuilder().CreateLoad(VarT, VarA);

  // Compute the end condition.
  // Convert condition to a bool by comparing non-equal to 0.0.
  EndV = getBuilder().CreateLoad(VarT, EndA);
  EndV = getBuilder().CreateICmpSLT(CurV, EndV, "loopcond");


  // Insert the conditional branch into the end of LoopEndBB.
  getBuilder().CreateCondBr(EndV, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(LoopBB);
  getBuilder().SetInsertPoint(LoopBB);
  
  //----------------------------------------------------------------------------
  // Set accessor

  for (const auto & AccessorPair : AccessorData) {
    auto i = AccessorPair.first;
    const auto & Data = AccessorPair.second;
    auto AccessorA = Data.first;
    auto PartA = Data.second;
    auto VarV = TheHelper_.load(VarA);
    TheHelper_.callFunction(
        TheModule,
        "contra_serial_accessor_setup",
        VoidType_,
        {VarV, PartA, ArgAs[i], AccessorA});
    ArgAs[i] = AccessorA;
  }

  //----------------------------------------------------------------------------
  // Call function

  ArgAs.emplace_back(VarA);

  std::vector<Value*> ArgVs;
  for (auto ArgA : ArgAs)
    ArgVs.emplace_back( TheHelper_.getAsValue(ArgA) );

  //------------------------------------
  // Call function with reduction
  if (ResultA && AbstractReduceOp) {
    auto ResultT = ResultA->getAllocatedType();
    auto ResultV = TheHelper_.callFunction(
        TheModule,
        TaskI.getName(),
        ResultT,
        ArgVs);
    auto ReduceOp = dynamic_cast<const SerialReduceInfo*>(AbstractReduceOp);
    auto NumReduce = ReduceOp->getNumReductions();
    for (unsigned i=0; i<NumReduce; ++i) {
      auto VarV = TheHelper_.extractValue(ResultV, i);
      auto ReduceV = TheHelper_.extractValue(ResultA, i);
      auto Op = ReduceOp->getReduceOp(i);
      ReduceV = applyReduce(TheModule, ReduceV, VarV, Op);
      TheHelper_.insertValue(ResultA, ReduceV, i);
    }
  }
  //------------------------------------
  // Call function without reduction
  else {
    TheHelper_.callFunction(
        TheModule,
        TaskI.getName(),
        VoidType_,
        ArgVs);
  }
  
  // Done loop
  //----------------------------------------------------------------------------

  // Insert unconditional branch to increment.
  getBuilder().CreateBr(IncrBB);
  
  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  getBuilder().SetInsertPoint(IncrBB);
  

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  TheHelper_.increment( TheHelper_.getAsAlloca(VarA), StepA );

  // Insert the conditional branch into the end of LoopEndBB.
  getBuilder().CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  getBuilder().SetInsertPoint(AfterBB);


  //----------------------------------------------------------------------------
  // cleanup
  
  destroyPartitions(TheModule, TempParts);
  destroyPartitionInfo(TheModule, PartInfoA);

  return ResultA;
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* SerialTasker::createPartition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* Color)
{

  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);
    
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);

  //------------------------------------
  if (isRange(Color)) {
    auto ColorA = TheHelper_.getAsAlloca(Color);

    std::vector<Value*> FunArgVs = {
      ColorA,
      IndexSpaceA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_serial_partition_from_index_space",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------
  else if (librt::DopeVector::isDopeVector(Color)) {
    auto ColorA = TheHelper_.getAsAlloca(Color);
  
    std::vector<Value*> FunArgVs = {
      ColorA,
      IndexSpaceA,
      IndexPartA,
      llvmValue<bool>(getContext(), true)
    };
    
    TheHelper_.callFunction(
        TheModule,
        "contra_serial_partition_from_array",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------
  else {
    auto ColorV = TheHelper_.getAsValue(Color);

    std::vector<Value*> FunArgVs = {
      ColorV,
      IndexSpaceA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_serial_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  
  return IndexPartA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* SerialTasker::createPartition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* IndexPartitionA,
    Value* ValueA)
{
  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);
    
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);

  //------------------------------------
  if (isField(ValueA)) {
    ValueA = TheHelper_.getAsAlloca(ValueA);
    IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionA);
    std::vector<Value*> FunArgVs = {
      ValueA,
      IndexSpaceA,
      IndexPartitionA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_serial_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
}

//==============================================================================
// Is this a field type
//==============================================================================
bool SerialTasker::isField(Value* FieldV) const
{
  auto FieldT = FieldV->getType();
  if (auto FieldA = dyn_cast<AllocaInst>(FieldV)) FieldT = FieldA->getAllocatedType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a serial field
//==============================================================================
void SerialTasker::createField(
    Module & TheModule,
    Value* FieldA,
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  auto NameV = llvmString(getContext(), TheModule, VarN);

  Value* DataSizeV;
  if (VarV) {
    DataSizeV = TheHelper_.getTypeSize<size_t>(VarT);
    VarV = TheHelper_.getAsAlloca(VarV);
  }
  else {
    DataSizeV = llvmValue<size_t>(getContext(), 0);
    VarV = Constant::getNullValue(VoidPtrType_);
  }
    
  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  std::vector<Value*> FunArgVs = {
    NameV,
    DataSizeV, 
    VarV,
    IndexSpaceA,
    FieldA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void SerialTasker::destroyField(Module &TheModule, Value* FieldA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_field_destroy",
      VoidType_,
      {FieldA});
}

//==============================================================================
// Is this an accessor type
//==============================================================================
bool SerialTasker::isAccessor(Type* AccessorT) const
{ return (AccessorT == AccessorType_); }

bool SerialTasker::isAccessor(Value* AccessorV) const
{
  auto AccessorT = AccessorV->getType();
  if (auto AccessorA = dyn_cast<AllocaInst>(AccessorV))
    AccessorT = AccessorA->getAllocatedType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void SerialTasker::storeAccessor(
    Module & TheModule,
    Value* ValueV,
    Value* AccessorV,
    Value* IndexV)
{
  auto ValueA = TheHelper_.getAsAlloca(ValueV);

  Value* AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  std::vector<Value*> FunArgVs = { AccessorA, ValueA };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(getContext(), 0) );
  }
  
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_accessor_write",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* SerialTasker::loadAccessor(
    Module & TheModule, 
    Type * ValueT,
    Value* AccessorV,
    Value* IndexV)
{
  auto AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  auto ValueA = TheHelper_.createEntryBlockAlloca(ValueT);

  std::vector<Value*> FunArgVs = { AccessorA, ValueA };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(getContext(), 0) );
  }

  TheHelper_.callFunction(
      TheModule,
      "contra_serial_accessor_read",
      VoidType_,
      FunArgVs);

  return ValueA;
}

//==============================================================================
// destroey an accessor
//==============================================================================
void SerialTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_accessor_destroy",
      VoidType_,
      {AccessorA});
}


//==============================================================================
// Is this an range type
//==============================================================================
bool SerialTasker::isPartition(Type* PartT) const
{ return (PartT == IndexPartitionType_); }

bool SerialTasker::isPartition(Value* PartV) const
{
  auto PartT = PartV->getType();
  if (auto PartA = dyn_cast<AllocaInst>(PartV)) PartT = PartA->getAllocatedType();
  return isPartition(PartT);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void SerialTasker::destroyPartition(
    Module &TheModule,
    Value* PartitionA)
{
  std::vector<Value*> FunArgVs = {PartitionA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_serial_partition_destroy",
      VoidType_,
      FunArgVs);
}
//==============================================================================
// create a reduction op
//==============================================================================
std::unique_ptr<AbstractReduceInfo> SerialTasker::createReductionOp(
    Module &,
    const std::string &,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{
  return std::make_unique<SerialReduceInfo>(VarTs, ReduceTypes);
}


}
