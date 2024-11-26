#include "mpi.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Support/raw_ostream.h"

#include <mpi.h>
  
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
  auto TheFunction = getBuilder().GetInsertBlock()->getParent();
  auto ThenBB = BasicBlock::Create(getContext(), "then", TheFunction);
  auto MergeBB = BasicBlock::Create(getContext(), "ifcont");
  getBuilder().CreateCondBr(CondV, ThenBB, MergeBB);
  getBuilder().SetInsertPoint(ThenBB);

  RootGuards_.push_front({MergeBB});

}

//==============================================================================
// Pop the root guard
//==============================================================================
void MpiTasker::popRootGuard(Module&)
{
  auto MergeBB = RootGuards_.front().MergeBlock;

  auto TheFunction = getBuilder().GetInsertBlock()->getParent();
  getBuilder().CreateBr(MergeBB);
  TheFunction->insert(TheFunction->end(), MergeBB);
  getBuilder().SetInsertPoint(MergeBB);
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
  auto NewType = StructType::create( getContext(), members, "contra_mpi_field_t" );
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
  auto NewType = StructType::create( getContext(), members, "contra_mpi_accessor_t" );
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
  auto NewType = StructType::create( getContext(), members, "contra_mpi_partition_t" );
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
  BasicBlock *BB = BasicBlock::Create(getContext(), "entry", WrapperF);
  getBuilder().SetInsertPoint(BB);
  
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
    getBuilder().CreateStore(&Arg, Alloca);
    WrapperArgAs.emplace_back(Alloca);
    ArgIdx++;
  }

  auto IndexA = WrapperArgAs.back();
  auto IndexT = IndexA->getAllocatedType();
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
  
  auto IndexV = getBuilder().CreateLoad(IndexT, IndexA);
  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    if (isRange(TaskArgTs[i])) {
      auto ArgN = TaskArgNs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      getBuilder().CreateCall(GetRangeF, {IndexV, WrapperArgAs[i], ArgA});
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
    // Have return value
    if (HasReturn) {
      ResultV = TheHelper_.getAsValue(ResultV);
      getBuilder().CreateRet(ResultV);
    }
    // No return value
    else {
      getBuilder().CreateRetVoid();
    }
    finishTask();
  }
  //----------------------------------------------------------------------------
  // Regular task
  else {
    // Have return value
    if (HasReturn) {
      ResultV = TheHelper_.getAsValue(ResultV);
      getBuilder().CreateRet(ResultV);
    }
    // No return value
    else {
      getBuilder().CreateRetVoid();
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
  getBuilder().CreateStore(IndexV, IndexA);

  Value* SizeV = TheHelper_.callFunction(
      TheModule,
      "contra_mpi_size",
      IntType_,
      {});
  auto SizeA = TheHelper_.createEntryBlockAlloca(IntType_, "size");
  getBuilder().CreateStore(SizeV, SizeA);
  
  //----------------------------------------------------------------------------
  // Determine loop bounds
  
  // Create an alloca for the variable in the entry block.
  auto VarT = IntType_;
  auto VarA = TheHelper_.createEntryBlockAlloca(VarT, "index");
  
  // Emit the start code first, without 'variable' in scope.
  auto EndA = TheHelper_.createEntryBlockAlloca(VarT, "end");
  auto StepA = TheHelper_.createEntryBlockAlloca(VarT, "step");

  auto IndexSpaceA = TheHelper_.getAsAlloca(RangeV);

  auto DistT = IntType_->getPointerTo();
  auto DistA = TheHelper_.createEntryBlockAlloca(DistT, "dist");
  SizeV = getBuilder().CreateLoad(IntType_, SizeA);
  auto OneC = llvmValue(getContext(), IntType_, 1);
  auto SizePlusOneV = getBuilder().CreateAdd(SizeV, OneC);
  auto IntSizeV = TheHelper_.getTypeSize<int_t>(IntType_);
  auto MallocSizeV = getBuilder().CreateMul( SizePlusOneV, IntSizeV );
  Value* DistV = TheHelper_.createMalloc(IntType_, MallocSizeV);
  getBuilder().CreateStore(DistV, DistA);

  DistV = getBuilder().CreateLoad(DistT, DistA);
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
        auto IndexPartitionV = getBuilder().CreateLoad(IndexPartitionType_, IndexPartitionPtr);
        IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionV);
      }


      FieldToPart[FieldA] = IndexPartitionA;

      DistV = getBuilder().CreateLoad(DistT, DistA);
      TheHelper_.callFunction(
          TheModule,
          "contra_mpi_field_fetch",
          VoidType_,
          {IndexSpaceA, DistV, IndexPartitionA, ArgAs[i]});

    } // field
  }
  
  //----------------------------------------------------------------------------
  // Reduction
  
  AllocaInst* ResultA = nullptr;

  if (AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const MpiReduceInfo*>(AbstractReduceOp);
    auto ResultT = StructType::create( getContext(), "reduce" );
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
  Value* EndV = getBuilder().CreateLoad(VarT, EndA);
  EndV = getBuilder().CreateICmpSLT(CurV, EndV, "loopcond");


  // Insert the conditional branch into the end of LoopEndBB.
  getBuilder().CreateCondBr(EndV, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(LoopBB);
  getBuilder().SetInsertPoint(LoopBB);
  
  //----------------------------------------------------------------------------
  // Call function
  
  CurV = getBuilder().CreateLoad(VarT, VarA);
  
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
   
  Type* ResultT = ResultA ? TheHelper_.getAllocatedType(ResultA) : VoidType_;

  auto ResultV = TheHelper_.callFunction(
      TheModule,
      TaskI.getName(),
      ResultT,
      {ArgVs});

  if (ResultA) {
    auto ReduceOp = dynamic_cast<const MpiReduceInfo*>(AbstractReduceOp);
    auto NumReduce = ReduceOp->getNumReductions();
    for (unsigned i=0; i<NumReduce; ++i) {
      auto VarV = getBuilder().CreateExtractValue(ResultV, i);
      auto ReduceV = TheHelper_.extractValue(ResultA, i);
      auto Op = ReduceOp->getReduceOp(i);
      ReduceV = applyReduce(TheModule, ReduceV, VarV, Op);
      TheHelper_.insertValue(ResultA, ReduceV, i);
    }
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
  TheHelper_.increment( VarA, StepA );

  // Insert the conditional branch into the end of LoopEndBB.
  getBuilder().CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  getBuilder().SetInsertPoint(AfterBB);
  
  //----------------------------------------------------------------------------
  // Reduction
  
  if (ResultA) {
    auto ReduceOp = dynamic_cast<const MpiReduceInfo*>(AbstractReduceOp);
    
    auto ResultT = TheHelper_.getAllocatedType(ResultA);
    auto TmpResultA = TheHelper_.createEntryBlockAlloca(ResultT);
    auto DataSizeV = llvmValue<size_t>(getContext(), ReduceOp->getDataSize()); 

    const auto & FoldN = ReduceOp->getFoldName();
    auto FoldT = ReduceOp->getFoldType();
    auto FoldF = TheModule.getOrInsertFunction(FoldN, FoldT).getCallee();

    TheHelper_.callFunction(
        TheModule,
        "contra_mpi_reduce",
        VoidType_,
        {FoldF, ResultA, TmpResultA, DataSizeV});

    ResultA = TmpResultA;
  }

  

  //----------------------------------------------------------------------------
  // cleanup
  
  DistV = getBuilder().CreateLoad(DistT, DistA);
  TheHelper_.createFree(DistV);
  
  destroyPartitions(TheModule, TempParts);
  destroyTaskInfo(TheModule, TaskInfoA);

  return ResultA;
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
      llvmValue<bool>(getContext(), true)
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
bool MpiTasker::isField(Value* FieldPtr) const
{
  auto FieldT = FieldPtr->getType();
  if (auto FieldA = dyn_cast<AllocaInst>(FieldPtr))
    FieldT = FieldA->getAllocatedType();
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

bool MpiTasker::isAccessor(Value* AccessorPtr) const
{
  auto AccessorT = AccessorPtr->getType();
  if (auto AccessorA = dyn_cast<AllocaInst>(AccessorPtr))
    AccessorT = AccessorA->getAllocatedType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void MpiTasker::storeAccessor(
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

bool MpiTasker::isPartition(Value* PartPtr) const
{
  auto PartT = PartPtr->getType();
  if (auto PartA = dyn_cast<AllocaInst>(PartPtr))
    PartT = PartA->getAllocatedType();
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
    Module & TheModule,
    const std::string & ReductionN,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{

  std::vector<Type*> ArgTs = {
    VoidPtrType_,
    VoidPtrType_,
    IntType_->getPointerTo(),
    llvmType<int>(getContext())->getPointerTo()
  };

  FunctionType* FunT = FunctionType::get(VoidType_, ArgTs, false);
  auto FunF = Function::Create(
      FunT,
      Function::ExternalLinkage,
      ReductionN,
      TheModule);

  auto BB = BasicBlock::Create(getContext(), "entry", FunF);
  getBuilder().SetInsertPoint(BB);
  
  auto InvecT = ArgTs[0];
  auto InvecPtrA = TheHelper_.createEntryBlockAlloca(ArgTs[0]);
  auto InoutvecT = ArgTs[1];
  auto InoutvecPtrA = TheHelper_.createEntryBlockAlloca(ArgTs[1]);

  auto ArgIt = FunF->arg_begin();
  getBuilder().CreateStore(ArgIt, InvecPtrA);
  ++ArgIt;
  getBuilder().CreateStore(ArgIt, InoutvecPtrA);

  std::size_t Offset = 0;
  for (unsigned i=0; i<VarTs.size(); ++i) {
    auto VarT = VarTs[i];
    auto VarPtrT = VarT->getPointerTo();
    auto DataSize = TheHelper_.getTypeSizeInBits(TheModule, VarT)/8;
    Value* InvecPtrV = getBuilder().CreateLoad(InvecT, InvecPtrA);
    InvecPtrV = TheHelper_.getElementPointer(InvecT, InvecPtrV, Offset);
    InvecPtrV = TheHelper_.createBitCast(InvecPtrV, VarPtrT);
    Value* InoutvecPtrV = getBuilder().CreateLoad(InoutvecT, InoutvecPtrA);
    InoutvecPtrV = TheHelper_.getElementPointer(InoutvecT, InoutvecPtrV, Offset);
    InoutvecPtrV = TheHelper_.createBitCast(InoutvecPtrV, VarPtrT);
    auto InvecV = getBuilder().CreateLoad(InvecT, InvecPtrV);
    auto InoutvecV = getBuilder().CreateLoad(InoutvecT, InoutvecPtrV);
    auto ResultV = foldReduce(TheModule, InvecV, InoutvecV, ReduceTypes[i]);
    getBuilder().CreateStore(ResultV, InoutvecPtrV);
    Offset += DataSize;
  }
  
  getBuilder().CreateRetVoid();


  return std::make_unique<MpiReduceInfo>(VarTs, ReduceTypes, FunF, Offset);
}


}
