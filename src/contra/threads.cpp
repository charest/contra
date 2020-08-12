#include "threads.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Support/raw_ostream.h"
  
////////////////////////////////////////////////////////////////////////////////
// Threads tasker
////////////////////////////////////////////////////////////////////////////////

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
// Constructor
//==============================================================================
ThreadsTasker::ThreadsTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  IndexSpaceType_ = DefaultIndexSpaceType_;
  IndexPartitionType_ = createIndexPartitionType();
  TaskInfoType_ = VoidPtrType_->getPointerTo();
  FieldType_ = createFieldType();
  AccessorType_ = createAccessorType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * ThreadsTasker::createFieldType()
{
  std::vector<Type*> members = {
    IntType_,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo() };
  auto NewType = StructType::create( TheContext_, members, "contra_threads_field_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * ThreadsTasker::createAccessorType()
{
  std::vector<Type*> members = {
    BoolType_,
    IntType_,
    VoidPtrType_};
  auto NewType = StructType::create( TheContext_, members, "contra_threads_accessor_t" );
  return NewType;
}


//==============================================================================
// Create the partition data type
//==============================================================================
StructType * ThreadsTasker::createIndexPartitionType()
{
  auto IntPtrT = IntType_->getPointerTo();
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    IntPtrT,
    IntPtrT,
    IndexSpaceType_->getPointerTo()};
  auto NewType = StructType::create( TheContext_, members, "contra_threads_partition_t" );
  return NewType;
}

//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* ThreadsTasker::createTaskInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(TaskInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_task_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void ThreadsTasker::destroyTaskInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_task_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// start runtime
//==============================================================================
void ThreadsTasker::startRuntime(Module &TheModule)
{
  launch(TheModule, *TopLevelTask_);
}

//==============================================================================
// Create the function wrapper
//==============================================================================
ThreadsTasker::PreambleResult ThreadsTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    llvm::Type* ResultT)
{

  startTask();

  auto WrapperT = FunctionType::get(VoidPtrType_, VoidPtrType_, false);
  auto WrapperF = Function::Create(
      WrapperT,
      Function::ExternalLinkage,
      TaskName,
      &TheModule);
  
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", WrapperF);
  Builder_.SetInsertPoint(BB);
  
  // store incoming arg
  auto Arg = WrapperF->arg_begin();
  Arg->setName("task_args");
  auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, Arg->getType(), "args");
  Builder_.CreateStore(Arg, ArgA);
  
  //----------------------------------------------------------------------------
  // Determine the thread data struct

  std::vector<Type*> ArgTs;
  ArgTs.reserve(TaskArgTs.size());
  
  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    auto ArgT = TaskArgTs[i];
    if (isAccessor(ArgT)) {
      ArgTs.emplace_back(FieldType_);
      ArgTs.emplace_back(IndexPartitionType_);
    }
    else if (isRange(ArgT)) {
      ArgTs.emplace_back(IndexPartitionType_);
    }
    else {
      ArgTs.emplace_back(ArgT);
    }
  }

  auto IndexLoc = ArgTs.size();
  ArgTs.emplace_back(IntType_);

  // reduction goes here
  if (ResultT) ArgTs.emplace_back(ResultT);

  auto ArgsT = StructType::create( TheContext_, ArgTs, "args_t" );
  Value* ArgsV = TheHelper_.load(ArgA);
  ArgsV = TheHelper_.createBitCast(ArgsV, ArgsT->getPointerTo());

  //----------------------------------------------------------------------------
  // extract index
  auto IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, IntType_, "index");
  {
    auto ArgA = TheHelper_.getElementPointer(ArgsV, 0, IndexLoc);
    auto ArgV = TheHelper_.load(ArgA);
    Builder_.CreateStore(ArgV, IndexA);
  }


  //----------------------------------------------------------------------------
  // extract result location
  if (ResultT) {
    auto & TaskE = getCurrentTask();
    TaskE.ResultAlloca = TheHelper_.getElementPointer(ArgsV, 0, ArgTs.size()-1);
  }
  
  //----------------------------------------------------------------------------
  // extract arguments
  
  std::vector<AllocaInst*> WrapperArgAs;
  WrapperArgAs.reserve(TaskArgTs.size());

  for (unsigned i=0, j=0; i<TaskArgNs.size(); ++i, ++j) {
    auto InArgA = TheHelper_.getElementPointer(ArgsV, 0, j);
    const auto & ArgN = TaskArgNs[i];
    auto ArgT = TaskArgTs[i];
    if (isRange(ArgT)) {
      auto IndexV = TheHelper_.load(IndexA);
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      TheHelper_.callFunction(
          TheModule,
          "contra_threads_index_space_create_from_partition",
          VoidType_,
          {IndexV, InArgA, ArgA});
      WrapperArgAs.emplace_back(ArgA);
    }
    else if (isAccessor(ArgT)) {
      auto IndexV = TheHelper_.load(IndexA);
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, AccessorType_, ArgN);
      auto InPartA = TheHelper_.getElementPointer(ArgsV, 0, ++j);
      TheHelper_.callFunction(
          TheModule,
          "contra_threads_accessor_setup",
          VoidType_,
          {IndexV, InPartA, InArgA, ArgA});
      WrapperArgAs.emplace_back(ArgA);
    }
    else {
      auto ArgV = TheHelper_.load(InArgA);
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgN);
      Builder_.CreateStore(ArgV, ArgA);
      WrapperArgAs.emplace_back(ArgA);
    }
  }

  return {WrapperF, WrapperArgAs, IndexA};
}

//==============================================================================
// Create the function wrapper
//==============================================================================
void ThreadsTasker::taskPostamble(
    Module &TheModule,
    Value* ResultV,
    bool IsIndex)
{
  bool HasReturn = (ResultV && !ResultV->getType()->isVoidTy());

  //----------------------------------------------------------------------------
  //  Index task
  if (IsIndex) {

    if (HasReturn) {
      ResultV = TheHelper_.getAsValue(ResultV);
      auto & TaskE = getCurrentTask();
      Builder_.CreateStore(ResultV, TaskE.ResultAlloca);
    }

    // always return null
    auto NullC = Constant::getNullValue(VoidPtrType_);
    Builder_.CreateRet(NullC);
    finishTask();
  }
  //----------------------------------------------------------------------------
  // Regular task
  else {
    // Have return value
    if (HasReturn) {
      ResultV = TheHelper_.getAsValue(ResultV);
      Builder_.CreateRet(ResultV);
    }
    // No return value
    else {
      Builder_.CreateRetVoid();
    }
  }

}
 

//==============================================================================
// Launch an index task
//==============================================================================
Value* ThreadsTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    std::vector<Value*> ArgAs,
    const std::vector<Value*> & PartAs,
    Value* RangeV,
    const AbstractReduceInfo* AbstractReduceOp)
{
  auto TaskInfoA = createTaskInfo(TheModule);

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
        TaskInfoA};
      TheHelper_.callFunction(
          TheModule,
          "contra_threads_register_index_partition",
          VoidType_,
          FunArgVs);
    }
  }

  //----------------------------------------------------------------------------
  // Prepare other args

  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);

  std::vector<Value*> ExpandedArgAs;
  ExpandedArgAs.reserve(NumArgs);

  for (unsigned i=0; i<NumArgs; i++) {
    ExpandedArgAs.emplace_back(ArgAs[i]);

    if (isField(ArgAs[i])) {

      auto FieldA = TheHelper_.getAsAlloca(ArgAs[i]);
      Value* IndexPartitionA = nullptr;
      if (PartAs[i]) {
        IndexPartitionA = TheHelper_.getAsAlloca(PartAs[i]);
      }
      else {
        auto IndexPartitionPtr = TheHelper_.callFunction(
            TheModule,
            "contra_threads_partition_get",
            IndexPartitionType_->getPointerTo(),
            {IndexSpaceA, FieldA, TaskInfoA});
        auto IndexPartitionV = TheHelper_.load(IndexPartitionPtr);
        IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionV);
      }

      ExpandedArgAs.emplace_back(IndexPartitionA);
    } // field
  }

  NumArgs = ExpandedArgAs.size();
  
  //----------------------------------------------------------------------------
  // Pack
  
  std::vector<Type*> ArgTs;
  for (auto A : ExpandedArgAs)
    ArgTs.emplace_back( TheHelper_.getAllocatedType(A) );

  ArgTs.emplace_back(IntType_); // index value

  AllocaInst* ResultA = nullptr;
  if (AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const ThreadsReduceInfo*>(AbstractReduceOp);
    auto ResultT = StructType::create( TheContext_, "reduce" );
    ResultT->setBody( ReduceOp->getVarTypes() );
    ArgTs.emplace_back(ResultT);
    ResultA = TheHelper_.createEntryBlockAlloca(ResultT);
  }
  
  auto ArgsT = StructType::create( TheContext_, ArgTs, "args_t" );
  auto ArgsSizeV = TheHelper_.getTypeSize<int_t>(ArgsT);
  auto RangeSizeV = getRangeSize(RangeV);
  auto SizeV = Builder_.CreateMul(ArgsSizeV, RangeSizeV); 
  auto MallocI = TheHelper_.createMalloc(ByteType_, SizeV, "args");
  auto ArgsA = TheHelper_.createEntryBlockAlloca(VoidPtrType_, "args.a");
  Builder_.CreateStore(MallocI, ArgsA);

  //----------------------------------------------------------------------------
  // create for loop
  
  // Create an alloca for the variable in the entry block.
  auto VarT = IntType_;
  auto VarA = TheHelper_.createEntryBlockAlloca(VarT, "index");
  
  // Emit the start code first, without 'variable' in scope.
  auto EndA = TheHelper_.createEntryBlockAlloca(VarT, "end");
  auto StepA = TheHelper_.createEntryBlockAlloca(VarT, "step");

  auto StartV = getRangeStart(RangeV);
  Builder_.CreateStore(StartV, VarA);
  auto EndV = getRangeEndPlusOne(RangeV);
  Builder_.CreateStore(EndV, EndA);
  auto StepV = getRangeStep(RangeV);
  Builder_.CreateStore(StepV, StepA);
  
  // Make the new basic block for the loop header, inserting after current
  // block.
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  BasicBlock *BeforeBB = BasicBlock::Create(TheContext_, "beforeloop", TheFunction);
  BasicBlock *LoopBB =   BasicBlock::Create(TheContext_, "loop", TheFunction);
  BasicBlock *IncrBB =   BasicBlock::Create(TheContext_, "incr", TheFunction);
  BasicBlock *AfterBB =  BasicBlock::Create(TheContext_, "afterloop", TheFunction);
  
  Builder_.CreateBr(BeforeBB);
  Builder_.SetInsertPoint(BeforeBB);

  // Load value and check coondition
  Value *CurV = TheHelper_.load(VarA);

  // Compute the end condition.
  // Convert condition to a bool by comparing non-equal to 0.0.
  EndV = TheHelper_.load(EndA);
  EndV = Builder_.CreateICmpSLT(CurV, EndV, "loopcond");


  // Insert the conditional branch into the end of LoopEndBB.
  Builder_.CreateCondBr(EndV, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(LoopBB);
  Builder_.SetInsertPoint(LoopBB);

  //----------------------------------------------------------------------------
  // Set thread args
  auto VarV = TheHelper_.load(VarA);
  Value* ThreadArgsV = TheHelper_.load(ArgsA);
  ThreadArgsV = TheHelper_.createBitCast(ThreadArgsV, ArgsT->getPointerTo());
  ThreadArgsV = TheHelper_.offsetPointer(ThreadArgsV, VarV);

  for (unsigned i=0; i<NumArgs; ++i) {
    auto ArgA = TheHelper_.getElementPointer(ThreadArgsV, {0, i});
    auto ArgV = TheHelper_.getAsValue(ExpandedArgAs[i]);
    Builder_.CreateStore(ArgV, ArgA);
  }

  { // index
    auto ArgA = TheHelper_.getElementPointer(ThreadArgsV, 0, NumArgs);
    auto VarV = TheHelper_.load(VarA); 
    Builder_.CreateStore(VarV, ArgA);
  }
  
  //----------------------------------------------------------------------------
  // Call function
  
  auto TaskF = TheModule.getOrInsertFunction(
      TaskI.getName(),
      TaskI.getFunctionType()).getCallee();

  TheHelper_.callFunction(
      TheModule,
      "contra_threads_launch",
      VoidType_,
      {TaskInfoA, TaskF, ThreadArgsV});
  
  // Done loop
  //----------------------------------------------------------------------------

  // Insert unconditional branch to increment.
  Builder_.CreateBr(IncrBB);
  
  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  Builder_.SetInsertPoint(IncrBB);
  

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  TheHelper_.increment( TheHelper_.getAsAlloca(VarA), StepA );

  // Insert the conditional branch into the end of LoopEndBB.
  Builder_.CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  Builder_.SetInsertPoint(AfterBB);

  // wait for threads
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_join",
      VoidType_,
      {TaskInfoA});
  
  //----------------------------------------------------------------------------
  // Apply reduction
  if (ResultA && AbstractReduceOp) {


    //----------------------------------
    // Init
    auto ReduceOp = dynamic_cast<const ThreadsReduceInfo*>(AbstractReduceOp);
    auto NumReduce = ReduceOp->getNumReductions();
    for (unsigned i=0; i<NumReduce; ++i) {
      auto VarT = ReduceOp->getVarType(i);
      auto Op = ReduceOp->getReduceOp(i);
      // get init value
      auto InitC = initReduce(VarT, Op);
      // store init
      TheHelper_.insertValue(ResultA, InitC, i);
    }
    
    //----------------------------------
    // Setup look for reduction
  
    auto StartV = getRangeStart(RangeV);
    Builder_.CreateStore(StartV, VarA);

    // Make the new basic block for the loop header, inserting after current
    // block.
    auto TheFunction = Builder_.GetInsertBlock()->getParent();
    BasicBlock *BeforeBB = BasicBlock::Create(TheContext_, "beforeloop", TheFunction);
    BasicBlock *LoopBB =   BasicBlock::Create(TheContext_, "loop", TheFunction);
    BasicBlock *IncrBB =   BasicBlock::Create(TheContext_, "incr", TheFunction);
    BasicBlock *AfterBB =  BasicBlock::Create(TheContext_, "afterloop", TheFunction);
    
    Builder_.CreateBr(BeforeBB);
    Builder_.SetInsertPoint(BeforeBB);

    // Load value and check coondition
    Value *CurV = TheHelper_.load(VarA);

    // Compute the end condition.
    // Convert condition to a bool by comparing non-equal to 0.0.
    auto EndV = TheHelper_.load(EndA);
    EndV = Builder_.CreateICmpSLT(CurV, EndV, "loopcond");


    // Insert the conditional branch into the end of LoopEndBB.
    Builder_.CreateCondBr(EndV, LoopBB, AfterBB);

    // Start insertion in LoopBB.
    //TheFunction->getBasicBlockList().push_back(LoopBB);
    Builder_.SetInsertPoint(LoopBB);

    //----------------------------------
    // Applly reduction
  
    auto VarV = TheHelper_.load(VarA);
    Value* ThreadArgsV = TheHelper_.load(ArgsA);
    ThreadArgsV = TheHelper_.createBitCast(ThreadArgsV, ArgsT->getPointerTo());
    ThreadArgsV = TheHelper_.offsetPointer(ThreadArgsV, VarV);
    auto ThreadResultA = TheHelper_.getElementPointer(ThreadArgsV, 0, NumArgs+1);
    
    for (unsigned i=0; i<NumReduce; ++i) {
      Value* VarV = TheHelper_.getElementPointer(ThreadResultA, 0, i);
      VarV = TheHelper_.load(VarV);
      auto ReduceV = TheHelper_.extractValue(ResultA, i);
      auto Op = ReduceOp->getReduceOp(i);
      ReduceV = applyReduce(TheModule, ReduceV, VarV, Op);
      TheHelper_.insertValue(ResultA, ReduceV, i);
    }
    
    //----------------------------------
    // Finish loop

    // Insert unconditional branch to increment.
    Builder_.CreateBr(IncrBB);
    
    // Start insertion in LoopBB.
    //TheFunction->getBasicBlockList().push_back(IncrBB);
    Builder_.SetInsertPoint(IncrBB);
    

    // Reload, increment, and restore the alloca.  This handles the case where
    // the body of the loop mutates the variable.
    TheHelper_.increment( TheHelper_.getAsAlloca(VarA), StepA );

    // Insert the conditional branch into the end of LoopEndBB.
    Builder_.CreateBr(BeforeBB);

    // Any new code will be inserted in AfterBB.
    //TheFunction->getBasicBlockList().push_back(AfterBB);
    Builder_.SetInsertPoint(AfterBB);
  }

  //----------------------------------------------------------------------------
  // cleanup
  
  destroyPartitions(TheModule, TempParts);
  destroyTaskInfo(TheModule, TaskInfoA);

  return ResultA;
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* ThreadsTasker::createPartition(
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
        "contra_threads_partition_from_index_space",
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
      llvmValue<bool>(TheContext_, true)
    };
    
    TheHelper_.callFunction(
        TheModule,
        "contra_threads_partition_from_array",
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
        "contra_threads_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  
  return IndexPartA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* ThreadsTasker::createPartition(
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
        "contra_threads_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
}

//==============================================================================
// Is this a field type
//==============================================================================
bool ThreadsTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a threaded field
//==============================================================================
void ThreadsTasker::createField(
    Module & TheModule,
    Value* FieldA,
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  auto NameV = llvmString(TheContext_, TheModule, VarN);

  Value* DataSizeV;
  if (VarV) {
    DataSizeV = TheHelper_.getTypeSize<size_t>(VarT);
    VarV = TheHelper_.getAsAlloca(VarV);
  }
  else {
    DataSizeV = llvmValue<size_t>(TheContext_, 0);
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
      "contra_threads_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void ThreadsTasker::destroyField(Module &TheModule, Value* FieldA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_field_destroy",
      VoidType_,
      {FieldA});
}

//==============================================================================
// Is this an accessor type
//==============================================================================
bool ThreadsTasker::isAccessor(Type* AccessorT) const
{ return (AccessorT == AccessorType_); }

bool ThreadsTasker::isAccessor(Value* AccessorA) const
{
  auto AccessorT = AccessorA->getType();
  if (isa<AllocaInst>(AccessorA)) AccessorT = AccessorT->getPointerElementType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void ThreadsTasker::storeAccessor(
    Module & TheModule,
    Value* ValueV,
    Value* AccessorV,
    Value* IndexV) const
{
  auto ValueA = TheHelper_.getAsAlloca(ValueV);

  Value* AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  std::vector<Value*> FunArgVs = { AccessorA, ValueA };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(TheContext_, 0) );
  }
  
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_accessor_write",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* ThreadsTasker::loadAccessor(
    Module & TheModule, 
    Type * ValueT,
    Value* AccessorV,
    Value* IndexV) const
{
  auto AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  auto ValueA = TheHelper_.createEntryBlockAlloca(ValueT);

  std::vector<Value*> FunArgVs = { AccessorA, ValueA };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(TheContext_, 0) );
  }

  TheHelper_.callFunction(
      TheModule,
      "contra_threads_accessor_read",
      VoidType_,
      FunArgVs);

  return ValueA;
}

//==============================================================================
// destroey an accessor
//==============================================================================
void ThreadsTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_accessor_destroy",
      VoidType_,
      {AccessorA});
}


//==============================================================================
// Is this an range type
//==============================================================================
bool ThreadsTasker::isPartition(Type* PartT) const
{ return (PartT == IndexPartitionType_); }

bool ThreadsTasker::isPartition(Value* PartA) const
{
  auto PartT = PartA->getType();
  if (isa<AllocaInst>(PartA)) PartT = PartT->getPointerElementType();
  return isPartition(PartT);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void ThreadsTasker::destroyPartition(
    Module &TheModule,
    Value* PartitionA)
{
  std::vector<Value*> FunArgVs = {PartitionA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_threads_partition_destroy",
      VoidType_,
      FunArgVs);
}
//==============================================================================
// create a reduction op
//==============================================================================
std::unique_ptr<AbstractReduceInfo> ThreadsTasker::createReductionOp(
    Module &,
    const std::string &,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{
  return std::make_unique<ThreadsReduceInfo>(VarTs, ReduceTypes);
}


}
