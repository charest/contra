#include "mpi.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Support/raw_ostream.h"
  
////////////////////////////////////////////////////////////////////////////////
// Mpi tasker
////////////////////////////////////////////////////////////////////////////////

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
// Constructor
//==============================================================================
MpiTasker::MpiTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  IndexSpaceType_ = DefaultIndexSpaceType_;
  IndexPartitionType_ = createIndexPartitionType();
  TaskInfoType_ = VoidPtrType_->getPointerTo();
  FieldType_ = createFieldType();
  AccessorType_ = createAccessorType();
}

//==============================================================================
// Mark a task
//==============================================================================
void MpiTasker::markTask(Module & M)
{
  TheHelper_.callFunction(
      M,
      "contra_mpi_mark_task",
      VoidType_);
}

//==============================================================================
// Unmark a task
//==============================================================================
void MpiTasker::unmarkTask(Module & M)
{
  TheHelper_.callFunction(
      M,
      "contra_mpi_unmark_task",
      VoidType_);
}

//==============================================================================
// Create a root guard
//==============================================================================
void MpiTasker::pushRootGuard(Module & M)
{
  auto TestV = TheHelper_.callFunction(
      M,
      "contra_mpi_test_root",
      BoolType_);
  auto CondV = TheHelper_.createCast(TestV, Int1Type_);

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto ThenBB = BasicBlock::Create(TheContext_, "then", TheFunction);
  auto MergeBB = BasicBlock::Create(TheContext_, "ifcont");
  Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
  Builder_.SetInsertPoint(ThenBB);

  RootGuards_.push_front({MergeBB});

}

//==============================================================================
// Pop the root guard
//==============================================================================
void MpiTasker::popRootGuard(Module&)
{
  auto MergeBB = RootGuards_.front().MergeBlock;

  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  Builder_.CreateBr(MergeBB);
  TheFunction->getBasicBlockList().push_back(MergeBB);
  Builder_.SetInsertPoint(MergeBB);
  RootGuards_.pop_front();
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * MpiTasker::createFieldType()
{
  std::vector<Type*> members = {
    IntType_,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo(),
    Int32Type_,
    IntType_->getPointerTo(),
    IndexPartitionType_->getPointerTo()
  };
  auto NewType = StructType::create( TheContext_, members, "contra_mpi_field_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * MpiTasker::createAccessorType()
{
  std::vector<Type*> members = {
    BoolType_,
    IntType_,
    VoidPtrType_};
  auto NewType = StructType::create( TheContext_, members, "contra_mpi_accessor_t" );
  return NewType;
}


//==============================================================================
// Create the partition data type
//==============================================================================
StructType * MpiTasker::createIndexPartitionType()
{
  auto IntPtrT = IntType_->getPointerTo();
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    IntPtrT,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo(),
    Int32Type_
  };
  auto NewType = StructType::create( TheContext_, members, "contra_mpi_partition_t" );
  return NewType;
}

//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* MpiTasker::createTaskInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(TaskInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_task_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void MpiTasker::destroyTaskInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_task_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// start runtime
//==============================================================================
void MpiTasker::startRuntime(Module &TheModule)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_init",
      VoidType_);

  launch(TheModule, *TopLevelTask_);
}

//==============================================================================
// Create the function wrapper
//==============================================================================
MpiTasker::PreambleResult MpiTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    llvm::Type* ResultT)
{

  startTask();
  
  //----------------------------------------------------------------------------
  // setup tasks
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
  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) Arg.setName(WrapperArgNs[ArgIdx++]);
  
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", WrapperF);
  Builder_.SetInsertPoint(BB);
  
  // allocate arguments
  std::vector<AllocaInst*> WrapperArgAs;
  WrapperArgAs.reserve(WrapperArgTs.size());

  ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgAs.emplace_back(Alloca);
    ArgIdx++;
  }
  
  auto IndexA = WrapperArgAs.back();
  WrapperArgAs.pop_back();
  
  //----------------------------------------------------------------------------
  // partition any ranges
  std::vector<Type*> GetRangeArgTs = {
    IntType_,
    IndexPartitionType_->getPointerTo(),
    IndexSpaceType_->getPointerTo()
  };

  auto GetRangeF = TheHelper_.createFunction(
      TheModule,
      "contra_mpi_index_space_create_from_partition",
      VoidType_,
      GetRangeArgTs);
  
  auto IndexV = TheHelper_.load(IndexA);
  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    if (isRange(TaskArgTs[i])) {
      auto ArgN = TaskArgNs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      Builder_.CreateCall(GetRangeF, {IndexV, WrapperArgAs[i], ArgA});
      WrapperArgAs[i] = ArgA;
    }
  }

  return {WrapperF, WrapperArgAs, IndexA};
}

//==============================================================================
// Create the function wrapper
//==============================================================================
void MpiTasker::taskPostamble(
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
Value* MpiTasker::launch(
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
          "contra_mpi_register_index_partition",
          VoidType_,
          FunArgVs);
    }
  }

  //----------------------------------------------------------------------------
  // extract index
  Value* IndexV = TheHelper_.callFunction(
      TheModule,
      "contra_mpi_rank",
      IntType_,
      {});
  auto IndexA = TheHelper_.createEntryBlockAlloca(IntType_, "index");
  Builder_.CreateStore(IndexV, IndexA);

  Value* SizeV = TheHelper_.callFunction(
      TheModule,
      "contra_mpi_size",
      IntType_,
      {});
  auto SizeA = TheHelper_.createEntryBlockAlloca(IntType_, "size");
  Builder_.CreateStore(SizeV, SizeA);
  
  //----------------------------------------------------------------------------
  // Determine loop bounds
  
  // Create an alloca for the variable in the entry block.
  auto VarT = IntType_;
  auto VarA = TheHelper_.createEntryBlockAlloca(VarT, "index");
  
  // Emit the start code first, without 'variable' in scope.
  auto EndA = TheHelper_.createEntryBlockAlloca(VarT, "end");
  auto StepA = TheHelper_.createEntryBlockAlloca(VarT, "step");

  auto IndexSpaceA = TheHelper_.getAsAlloca(RangeV);

  auto DistA = TheHelper_.createEntryBlockAlloca(IntType_->getPointerTo(), "dist");
  SizeV = TheHelper_.load(SizeA);
  auto OneC = llvmValue(TheContext_, IntType_, 1);
  auto SizePlusOneV = Builder_.CreateAdd(SizeV, OneC);
  auto IntSizeV = TheHelper_.getTypeSize<int_t>(IntType_);
  auto MallocSizeV = Builder_.CreateMul( SizePlusOneV, IntSizeV );
  Value* DistV = TheHelper_.createMalloc(IntType_, MallocSizeV);
  Builder_.CreateStore(DistV, DistA);

  DistV = TheHelper_.load(DistA);
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_loop_bounds",
      VoidType_,
      {IndexSpaceA, VarA, EndA, StepA, DistV});
  
  //----------------------------------------------------------------------------
  // Fetch fields
  
  std::map<Value*, Value*> FieldToPart;

  for (unsigned i=0; i<NumArgs; i++) {
    if (isField(ArgAs[i])) {

      auto FieldA = TheHelper_.getAsAlloca(ArgAs[i]);
      Value* IndexPartitionA = nullptr;
      if (PartAs[i]) {
        IndexPartitionA = TheHelper_.getAsAlloca(PartAs[i]);
      }
      else {
        auto IndexPartitionPtr = TheHelper_.callFunction(
            TheModule,
            "contra_mpi_partition_get",
            IndexPartitionType_->getPointerTo(),
            {IndexSpaceA, FieldA, TaskInfoA});
        auto IndexPartitionV = TheHelper_.load(IndexPartitionPtr);
        IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionV);
      }


      FieldToPart[FieldA] = IndexPartitionA;

      DistV = TheHelper_.load(DistA);
      TheHelper_.callFunction(
          TheModule,
          "contra_mpi_field_fetch",
          VoidType_,
          {IndexSpaceA, DistV, IndexPartitionA, ArgAs[i]});

    } // field
  }
  
  
  //----------------------------------------------------------------------------
  // create for loop
  
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
  
  //----------------------------------------------------------------------------
  // Call function
  
  CurV = TheHelper_.load(VarA);
  
  std::vector<Value*> ArgVs;
  for (auto ArgA : ArgAs) {
    if (isField(ArgA)) {
      auto AccA = TheHelper_.createEntryBlockAlloca(AccessorType_, "acc");
      auto PartA = FieldToPart.at(ArgA);
      TheHelper_.callFunction(
          TheModule,
          "contra_mpi_accessor_setup",
          VoidType_,
          {CurV, PartA, ArgA, AccA}); 
      ArgA = AccA;
    }
    ArgVs.emplace_back( TheHelper_.getAsValue(ArgA) );
  }
  
  ArgVs.emplace_back( CurV );
   
  TheHelper_.callFunction(
      TheModule,
      TaskI.getName(),
      VoidType_,
      {ArgVs});
  
  // Done loop
  //----------------------------------------------------------------------------

  // Insert unconditional branch to increment.
  Builder_.CreateBr(IncrBB);
  
  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  Builder_.SetInsertPoint(IncrBB);
  

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  TheHelper_.increment( VarA, StepA );

  // Insert the conditional branch into the end of LoopEndBB.
  Builder_.CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  Builder_.SetInsertPoint(AfterBB);

  //----------------------------------------------------------------------------
  // cleanup
  
  DistV = TheHelper_.load(DistA);
  TheHelper_.createFree(DistV);
  
  destroyPartitions(TheModule, TempParts);
  destroyTaskInfo(TheModule, TaskInfoA);

  return nullptr; //ResultA;
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* MpiTasker::createPartition(
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
        "contra_mpi_partition_from_index_space",
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
        "contra_mpi_partition_from_array",
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
        "contra_mpi_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  
  return IndexPartA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* MpiTasker::createPartition(
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
        "contra_mpi_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
}

//==============================================================================
// Is this a field type
//==============================================================================
bool MpiTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a threaded field
//==============================================================================
void MpiTasker::createField(
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
      "contra_mpi_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void MpiTasker::destroyField(Module &TheModule, Value* FieldA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_field_destroy",
      VoidType_,
      {FieldA});
}

//==============================================================================
// Is this an accessor type
//==============================================================================
bool MpiTasker::isAccessor(Type* AccessorT) const
{ return (AccessorT == AccessorType_); }

bool MpiTasker::isAccessor(Value* AccessorA) const
{
  auto AccessorT = AccessorA->getType();
  if (isa<AllocaInst>(AccessorA)) AccessorT = AccessorT->getPointerElementType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void MpiTasker::storeAccessor(
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
      "contra_mpi_accessor_write",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* MpiTasker::loadAccessor(
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
      "contra_mpi_accessor_read",
      VoidType_,
      FunArgVs);

  return ValueA;
}

//==============================================================================
// destroey an accessor
//==============================================================================
void MpiTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_accessor_destroy",
      VoidType_,
      {AccessorA});
}


//==============================================================================
// Is this an range type
//==============================================================================
bool MpiTasker::isPartition(Type* PartT) const
{ return (PartT == IndexPartitionType_); }

bool MpiTasker::isPartition(Value* PartA) const
{
  auto PartT = PartA->getType();
  if (isa<AllocaInst>(PartA)) PartT = PartT->getPointerElementType();
  return isPartition(PartT);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void MpiTasker::destroyPartition(
    Module &TheModule,
    Value* PartitionA)
{
  std::vector<Value*> FunArgVs = {PartitionA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_mpi_partition_destroy",
      VoidType_,
      FunArgVs);
}
//==============================================================================
// create a reduction op
//==============================================================================
std::unique_ptr<AbstractReduceInfo> MpiTasker::createReductionOp(
    Module &,
    const std::string &,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{
  return std::make_unique<MpiReduceInfo>(VarTs, ReduceTypes);
}


}
