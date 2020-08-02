#include "cuda.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/raw_ostream.h"
  
////////////////////////////////////////////////////////////////////////////////
// Legion tasker
////////////////////////////////////////////////////////////////////////////////

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
// Constructor
//==============================================================================
CudaTasker::CudaTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  IndexSpaceType_ = DefaultIndexSpaceType_;
  IndexPartitionType_ = createIndexPartitionType();
  PartitionInfoType_ = VoidPtrType_->getPointerTo();
  TaskInfoType_ = VoidPtrType_->getPointerTo();
  FieldType_ = createFieldType();
  AccessorType_ = createAccessorType();
}

//==============================================================================
// Create the partition data type
//==============================================================================
StructType * CudaTasker::createIndexPartitionType()
{
  auto IntPtrT = IntType_->getPointerTo();
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    IntPtrT,
    IntPtrT};
  auto NewType = StructType::create( TheContext_, members, "contra_cuda_partition_t" );
  return NewType;
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * CudaTasker::createFieldType()
{
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo()};
  auto NewType = StructType::create( TheContext_, members, "contra_cuda_field_t" );
  return NewType;
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * CudaTasker::createAccessorType()
{
  std::vector<Type*> members = {
    IntType_,
    VoidPtrType_,
    IndexPartitionType_,
    FieldType_->getPointerTo() };
  auto NewType = StructType::create( TheContext_, members, "contra_cuda_accessor_t" );
  return NewType;
}


//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* CudaTasker::createPartitionInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(PartitionInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_cuda_partition_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void CudaTasker::destroyPartitionInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_cuda_partition_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* CudaTasker::createTaskInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(TaskInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_cuda_task_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void CudaTasker::destroyTaskInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_cuda_task_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// start runtime
//==============================================================================
void CudaTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{
  //TheHelper_.callFunction(
  //    TheModule,
  //    "contra_cuda_startup",
  //    VoidType_);

  launch(TheModule, *TopLevelTask_);
}
//==============================================================================
// Default Preamble
//==============================================================================
CudaTasker::PreambleResult CudaTasker::taskPreamble(
    Module &TheModule,
    const std::string & Name,
    Function* TaskF)
{
  startTask();
  return AbstractTasker::taskPreamble(TheModule, Name, TaskF);
}

//==============================================================================
// Create the function wrapper
//==============================================================================
CudaTasker::PreambleResult CudaTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    llvm::Type* ResultT)
{

  auto & TaskI = startTask();

  //----------------------------------------------------------------------------
  // Create function header
  std::vector<std::string> WrapperArgNs = TaskArgNs;
    
  std::vector<Type*> WrapperArgTs;
  for (auto ArgT : TaskArgTs) {
    if (isRange(ArgT)) ArgT = IndexPartitionType_;
    WrapperArgTs.emplace_back(ArgT);
  }
    
  WrapperArgTs.emplace_back(IntType_);
  WrapperArgNs.emplace_back("size");

  if (ResultT) {
    WrapperArgTs.emplace_back(VoidPtrType_);
    WrapperArgNs.emplace_back("indata");
  }

  auto WrapperT = FunctionType::get(VoidType_, WrapperArgTs, false);
  auto WrapperF = Function::Create(
      WrapperT,
      Function::ExternalLinkage,
      TaskName,
      &TheModule);

  // annotate as kernel
  auto Annots = TheModule.getOrInsertNamedMetadata("nvvm.annotations");
  std::vector<Metadata*> Meta = {
    ValueAsMetadata::get(WrapperF),
    MDString::get(TheContext_, "kernel"),
    ValueAsMetadata::get( llvmValue<int>(TheContext_, 1) ) };
  Annots->addOperand(MDNode::get(TheContext_, Meta));
  
  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : WrapperF->args()) Arg.setName(WrapperArgNs[Idx++]);
  
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", WrapperF);
  Builder_.SetInsertPoint(BB);
  
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
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgAs.emplace_back(Alloca);
    ArgIdx++;
  }
  
  //----------------------------------------------------------------------------
  // determine my index
 
  // tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto TidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::nvvm_read_ptx_sreg_tid_x);
  Value* IndexV = Builder_.CreateCall(TidF);
  auto BidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  Value* BlockIdV = Builder_.CreateCall(BidF);
  auto BdimF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  Value* BlockDimV = Builder_.CreateCall(BdimF);

  Value* TmpV = Builder_.CreateMul(BlockIdV, BlockDimV);
  IndexV = Builder_.CreateAdd(IndexV, TmpV);

  // cast and store
  IndexV = TheHelper_.createCast(IndexV, IntType_);
  auto IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, IntType_, "index");
  Builder_.CreateStore(IndexV, IndexA);
  
  //----------------------------------------------------------------------------
  // If tid < total size

  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  BasicBlock *ThenBB = BasicBlock::Create(TheContext_, "then", TheFunction);
  TaskI.MergeBlock = BasicBlock::Create(TheContext_, "ifcont");

  auto IndexSizeA = WrapperArgAs[TaskArgNs.size()];
  auto IndexSizeV = TheHelper_.load(IndexSizeA);

  IndexV = TheHelper_.load(IndexA); 
  auto CondV = Builder_.CreateICmpSLT(IndexV, IndexSizeV, "threadcond");
  Builder_.CreateCondBr(CondV, ThenBB, TaskI.MergeBlock);
  
  // Emit then value.
  Builder_.SetInsertPoint(ThenBB);

  //----------------------------------------------------------------------------
  // partition any ranges
  std::vector<Type*> GetRangeArgTs = {
    IntType_,
    IndexPartitionType_->getPointerTo(),
    IndexSpaceType_->getPointerTo()
  };

  auto GetRangeF = TheHelper_.createFunction(
      TheModule,
      "contra_cuda_index_space_create_from_partition",
      VoidType_,
      GetRangeArgTs);
  
  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    if (isRange(TaskArgTs[i])) {
      auto ArgN = TaskArgNs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      Builder_.CreateCall(GetRangeF, {IndexV, WrapperArgAs[i], ArgA});
      WrapperArgAs[i] = ArgA;
    }
  }

  if (ResultT) {
    TaskI.ResultAlloca = WrapperArgAs.back();
    WrapperArgAs.pop_back();
  }

  // Index size
  WrapperArgAs.pop_back();
  
  return {WrapperF, WrapperArgAs, IndexA};
}

//==============================================================================
// Create the function wrapper
//==============================================================================
void CudaTasker::taskPostamble(
    Module &TheModule,
    Value* ResultV,
    bool IsIndex)
{
      
  auto & TaskI = getCurrentTask();
  
  // destroy existing task info if it was created
  if (TaskI.TaskInfoAlloca) destroyTaskInfo(TheModule, TaskI.TaskInfoAlloca);
  TaskI.TaskInfoAlloca = nullptr;

  //----------------------------------------------------------------------------
  // Index task
  if (IsIndex) {

    // Call the reduction
    if (ResultV && !ResultV->getType()->isVoidTy()) {
      auto IndataA = TaskI.ResultAlloca;
      
      auto ResultA = TheHelper_.getAsAlloca(ResultV);
      auto ResultT = ResultA->getAllocatedType();
      
      auto DataSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
      
      TheHelper_.callFunction(
          TheModule,
          "contra_cuda_set_reduction_value",
          VoidType_,
          {IndataA, ResultA, DataSizeV});
    }
  
    // finish If
    auto MergeBB = TaskI.MergeBlock;
    Builder_.CreateBr(MergeBB);
    auto TheFunction = Builder_.GetInsertBlock()->getParent();
    TheFunction->getBasicBlockList().push_back(MergeBB);
    Builder_.SetInsertPoint(MergeBB);

    Builder_.CreateRetVoid();
  }
  //----------------------------------------------------------------------------
  // Single task
  else {

    // Have return value
    if (ResultV && !ResultV->getType()->isVoidTy()) {
      ResultV = TheHelper_.getAsValue(ResultV);
      Builder_.CreateRet(ResultV);
    }
    // no return value
    else {
      Builder_.CreateRetVoid();
    }

  }

  //----------------------------------------------------------------------------
  // Finish 

  finishTask();
}
 

//==============================================================================
// Launch an index task
//==============================================================================
Value* CudaTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    std::vector<Value*> ArgAs,
    const std::vector<Value*> & PartAs,
    Value* RangeV,
    const AbstractReduceInfo* AbstractReduceOp)
{

  std::vector<AllocaInst*> ToFree;
  
  auto & TaskE = getCurrentTask();
  auto & TaskInfoA = TaskE.TaskInfoAlloca;
  if (!TaskInfoA) TaskInfoA = createTaskInfo(TheModule);


  //----------------------------------------------------------------------------
  // Swap ranges for partitions
  
  auto PartInfoA = createPartitionInfo(TheModule);
  
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
        PartInfoA,
        TaskInfoA};
      auto DevIndexPartitionPtr = TheHelper_.callFunction(
          TheModule,
          "contra_cuda_partition2dev",
          IndexPartitionType_,
          FunArgVs);
      auto DevIndexPartitionA = TheHelper_.getAsAlloca(DevIndexPartitionPtr);
      if (!PartAs[i]) ToFree.emplace_back(DevIndexPartitionA);
      ArgAs[i] = DevIndexPartitionA;
    }
  }
  
  //----------------------------------------------------------------------------
  // Setup args
  
  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  for (unsigned i=0; i<NumArgs; i++) {
    auto ArgA = ArgAs[i];

    //----------------------------------
    // Array
    if (librt::DopeVector::isDopeVector(ArgA)) {
      auto ArrayA = TheHelper_.getAsAlloca(ArgA);
      auto ArrayT = librt::DopeVector::DopeVectorType;
      auto DevPtrA = TheHelper_.createEntryBlockAlloca(ArrayT);
      TheHelper_.callFunction(
          TheModule,
          "contra_cuda_array2dev",
          VoidType_,
          {ArrayA, DevPtrA});
      ToFree.emplace_back( DevPtrA );
      ArgAs[i] = DevPtrA;
    }
    //----------------------------------
    // Field
    else if (isField(ArgA)) {
      auto FieldA = TheHelper_.getAsAlloca(ArgA);
      Value* PartA = Constant::getNullValue(IndexPartitionType_->getPointerTo());
      if (PartAs[i]) PartA = TheHelper_.getAsAlloca(PartAs[i]);
      auto DevFieldV = TheHelper_.callFunction(
            TheModule,
            "contra_cuda_field2dev",
            AccessorType_,
            {IndexSpaceA, PartA, FieldA, PartInfoA, TaskInfoA});
      auto DevFieldA = TheHelper_.getAsAlloca(DevFieldV);
      ToFree.emplace_back( DevFieldA );
      ArgAs[i] = DevFieldA;
    }
    //----------------------------------
    // Scalar
    else {
      ArgAs[i] = TheHelper_.getAsAlloca(ArgA);
    }
  }

  // add index size
  auto IndexSpaceSizeA = TheHelper_.createEntryBlockAlloca(IntType_);
  Builder_.CreateStore( getRangeSize(IndexSpaceA), IndexSpaceSizeA );
  ArgAs.emplace_back( IndexSpaceSizeA );
  
  //----------------------------------------------------------------------------
  // Do reductions
  AllocaInst* IndataA = nullptr;
  StructType* ResultT = nullptr;
  
  if (AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const CudaReduceInfo*>(AbstractReduceOp);
    
    ResultT = StructType::create( TheContext_, "reduce" );
    ResultT->setBody( ReduceOp->getVarTypes() );
    IndataA = TheHelper_.createEntryBlockAlloca(VoidPtrType_);
    auto DataSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
    TheHelper_.callFunction(
        TheModule,
        "contra_cuda_prepare_reduction",
        VoidType_,
        {IndataA, DataSizeV, IndexSpaceA});
    ToFree.emplace_back(IndataA);
    ArgAs.emplace_back( IndataA );
  }
  
  //----------------------------------------------------------------------------
  // Serialize args
  
  NumArgs = ArgAs.size();
  auto ArgsT = ArrayType::get(VoidPtrType_, NumArgs);
  auto ArgsA = TheHelper_.createEntryBlockAlloca(ArgsT);

  for (unsigned i=0; i<NumArgs; ++i) {
    auto ArgV = TheHelper_.createBitCast(ArgAs[i], VoidPtrType_);
    TheHelper_.insertValue(ArgsA, ArgV, i);
  }
    
  auto RangeA = TheHelper_.getAsAlloca(RangeV);
  

  //----------------------------------------------------------------------------
  // Call function
  auto TaskStr = llvmString(TheContext_, TheModule, TaskI.getName());
  TheHelper_.callFunction(
    TheModule,
    "contra_cuda_launch_kernel",
    VoidType_,
    {TaskStr, RangeA, ArgsA});

  //----------------------------------------------------------------------------
  // Call function with reduction
  AllocaInst* ResultA = nullptr;
  if (ResultT && AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const CudaReduceInfo*>(AbstractReduceOp);

    auto InitStr = llvmString(TheContext_, TheModule, ReduceOp->getInitPtrName());
    auto ApplyStr = llvmString(TheContext_, TheModule, ReduceOp->getApplyPtrName());
    auto FoldStr = llvmString(TheContext_, TheModule, ReduceOp->getFoldPtrName());

    const auto & FoldN = ReduceOp->getFoldName();
    auto FoldT = ReduceOp->getFoldType();
    auto FoldF = TheModule.getOrInsertFunction(FoldN, FoldT).getCallee();

    ResultA = TheHelper_.createEntryBlockAlloca(ResultT);
    auto OutDataV = TheHelper_.createBitCast(ResultA, VoidPtrType_);
    auto DataSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
    std::vector<Value*> ArgVs = {
      TaskStr,
      InitStr,
      ApplyStr,
      FoldStr,
      IndexSpaceA,
      IndataA,
      OutDataV,
      DataSizeV,
      FoldF
    };
    TheHelper_.callFunction(
        TheModule,
        "contra_cuda_launch_reduction",
        VoidType_,
        ArgVs);
  }
  
  //----------------------------------------------------------------------------
  // cleanup

  for (auto AllocA : ToFree) {
    if (isPartition(AllocA)) {
      TheHelper_.callFunction(
          TheModule,
          "contra_cuda_partition_free_and_deregister",
          VoidType_,
          {AllocA, TaskInfoA});
    }
    else if (isAccessor(AllocA)) {
      TheHelper_.callFunction(
          TheModule,
          "contra_cuda_accessor_free_temp",
          VoidType_,
          {AllocA, TaskInfoA});
    }
    else if (librt::DopeVector::isDopeVector(AllocA)) {
      TheHelper_.callFunction(
          TheModule,
          "contra_cuda_array_free",
          VoidType_,
          {AllocA});
    }
    else {
      TheHelper_.callFunction(
          TheModule,
          "contra_cuda_free",
          VoidType_,
          {AllocA});
    }
  }
  
  destroyPartitions(TheModule, TempParts);
  destroyPartitionInfo(TheModule, PartInfoA);
  
  return ResultA;
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* CudaTasker::createPartition(
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
        "contra_cuda_partition_from_index_space",
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
        "contra_cuda_partition_from_array",
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
        "contra_cuda_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  
  return IndexPartA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* CudaTasker::createPartition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* IndexPartitionA,
    Value* ValueA)
{
  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);
    
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);

  //------------------------------------
  if (isField(ValueA)) {
    const auto & TaskI = getCurrentTask();
   
    ValueA = TheHelper_.getAsAlloca(ValueA);
    IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionA);
    std::vector<Value*> FunArgVs = {
      ValueA,
      IndexSpaceA,
      IndexPartitionA,
      IndexPartA,
      TaskI.TaskInfoAlloca};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_cuda_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
}

//==============================================================================
// Is this a field type
//==============================================================================
bool CudaTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a legion field
//==============================================================================
void CudaTasker::createField(
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
      "contra_cuda_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void CudaTasker::destroyField(Module &TheModule, Value* FieldA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_cuda_field_destroy",
      VoidType_,
      {FieldA});
}

//==============================================================================
// Is this an accessor type
//==============================================================================
bool CudaTasker::isAccessor(Type* AccessorT) const
{ return (AccessorT == AccessorType_); }

bool CudaTasker::isAccessor(Value* AccessorA) const
{
  auto AccessorT = AccessorA->getType();
  if (isa<AllocaInst>(AccessorA)) AccessorT = AccessorT->getPointerElementType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void CudaTasker::storeAccessor(
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
      "contra_cuda_accessor_write",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* CudaTasker::loadAccessor(
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
      "contra_cuda_accessor_read",
      VoidType_,
      FunArgVs);

  return ValueA;
}

//==============================================================================
// destroey an accessor
//==============================================================================
void CudaTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  //TheHelper_.callFunction(
  //    TheModule,
  //    "contra_cuda_accessor_destroy",
  //    VoidType_,
  //    {AccessorA});
}

//==============================================================================
// Is this an range type
//==============================================================================
bool CudaTasker::isPartition(Type* PartT) const
{ return (PartT == IndexPartitionType_); }

bool CudaTasker::isPartition(Value* PartA) const
{
  auto PartT = PartA->getType();
  if (isa<AllocaInst>(PartA)) PartT = PartT->getPointerElementType();
  return isPartition(PartT);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void CudaTasker::destroyPartition(
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
std::unique_ptr<AbstractReduceInfo> CudaTasker::createReductionOp(
    Module & TheModule,
    const std::string &ReductionN,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{

  // get var types

  // get data size
  std::size_t DataSize = 0;
  std::vector<std::size_t> DataSizes;
  for (auto VarT : VarTs) {
    DataSizes.emplace_back( TheHelper_.getTypeSizeInBits(TheModule, VarT)/8 );
    DataSize += DataSizes.back();
  }

  //----------------------------------------------------------------------------
  // create fold
  Function * FoldF;
  std::string FoldPtrN;
  {
    std::string FoldN = ReductionN + "fold";

    std::vector<Type*> ArgTs = {
      VoidPtrType_,
      VoidPtrType_
    };
    FunctionType* FunT = FunctionType::get(VoidType_, ArgTs, false);
    FoldF = Function::Create(
        FunT,
        Function::ExternalLinkage,
        FoldN,
        TheModule);

    auto BB = BasicBlock::Create(TheContext_, "entry", FoldF);
    Builder_.SetInsertPoint(BB);

    unsigned i=0;
    std::vector<AllocaInst*> ArgAs(ArgTs.size());
    for (auto &Arg : FoldF->args()) {
      ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
      Builder_.CreateStore(&Arg, ArgAs[i]);
      ++i;
    }

    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    auto ZeroC = llvmValue(TheContext_, SizeType_, 0);
    Builder_.CreateStore(ZeroC, OffsetA);


    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      Value* RhsPtrV = TheHelper_.load(ArgAs[1]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = Builder_.CreateGEP(LhsPtrV, OffsetV);
      RhsPtrV = Builder_.CreateGEP(RhsPtrV, OffsetV);
      auto VarT = VarTs[i];
      auto VarPtrT = VarT->getPointerTo();
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarPtrT);
      RhsPtrV = TheHelper_.createBitCast(RhsPtrV, VarPtrT);
      auto LhsV = Builder_.CreateLoad(VarT, LhsPtrV, true /*volatile*/);
      auto RhsV = Builder_.CreateLoad(VarT, RhsPtrV, true /*volatile*/);
      auto ReduceV = foldReduce(TheModule, LhsV, RhsV, ReduceTypes[i]);
      Builder_.CreateStore(ReduceV, LhsPtrV, true /*volatile*/);
      auto SizeC = llvmValue(TheContext_, SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
        
    Builder_.CreateRetVoid();
    
    // device pointer
    FoldPtrN = FoldN + "_ptr";
    new GlobalVariable(
        TheModule, 
        FunT->getPointerTo(),
        false,
        GlobalValue::InternalLinkage,
        FoldF, // has initializer, specified below
        FoldPtrN,
        nullptr,
        GlobalValue::NotThreadLocal,
        1);
  }


  //----------------------------------------------------------------------------
  // create apply + fold operation
  Function * ApplyF;
  std::string ApplyPtrN;
  {
    std::string ApplyN = ReductionN + "apply";

    std::vector<Type*> ArgTs = {
      VoidPtrType_,
      VoidPtrType_
    };
    FunctionType* FunT = FunctionType::get(VoidType_, ArgTs, false);
    ApplyF = Function::Create(
        FunT,
        Function::ExternalLinkage,
        ApplyN,
        TheModule);

    auto BB = BasicBlock::Create(TheContext_, "entry", ApplyF);
    Builder_.SetInsertPoint(BB);

    unsigned i=0;
    std::vector<AllocaInst*> ArgAs(ArgTs.size());
    for (auto &Arg : ApplyF->args()) {
      ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
      Builder_.CreateStore(&Arg, ArgAs[i]);
      ++i;
    }

    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    auto ZeroC = llvmValue(TheContext_, SizeType_, 0);
    Builder_.CreateStore(ZeroC, OffsetA);


    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      Value* RhsPtrV = TheHelper_.load(ArgAs[1]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = Builder_.CreateGEP(LhsPtrV, OffsetV);
      RhsPtrV = Builder_.CreateGEP(RhsPtrV, OffsetV);
      auto VarT = VarTs[i];
      auto VarPtrT = VarT->getPointerTo();
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarPtrT);
      RhsPtrV = TheHelper_.createBitCast(RhsPtrV, VarPtrT);
      auto LhsV = Builder_.CreateLoad(VarT, LhsPtrV, true /*volatile*/);
      auto RhsV = Builder_.CreateLoad(VarT, RhsPtrV, true /*volatile*/);
      auto ReduceV = applyReduce(TheModule, LhsV, RhsV, ReduceTypes[i]);
      Builder_.CreateStore(ReduceV, LhsPtrV, true /*volatile*/);
      auto SizeC = llvmValue(TheContext_, SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
        
    Builder_.CreateRetVoid();
    
    // device pointer
    ApplyPtrN = ApplyN + "_ptr";
    new GlobalVariable(
        TheModule, 
        FunT->getPointerTo(),
        false,
        GlobalValue::InternalLinkage,
        ApplyF, // has initializer, specified below
        ApplyPtrN,
        nullptr,
        GlobalValue::NotThreadLocal,
        1);
  }

  //----------------------------------------------------------------------------
  // create init
  Function * InitF;
  std::string InitPtrN;
  {

    std::string InitN = ReductionN + "init";
 
    std::vector<Type*> ArgTs = {VoidPtrType_};
    FunctionType* InitT = FunctionType::get(VoidType_, ArgTs, false);
    InitF = Function::Create(
        InitT,
        Function::ExternalLinkage,
        InitN,
        TheModule);
  
    auto BB = BasicBlock::Create(TheContext_, "entry", InitF);
    Builder_.SetInsertPoint(BB);
    
    unsigned i=0;
    std::vector<AllocaInst*> ArgAs(ArgTs.size());
    for (auto &Arg : InitF->args()) {
      ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
      Builder_.CreateStore(&Arg, ArgAs[i]);
      ++i;
    }

    auto ZeroC = llvmValue(TheContext_, SizeType_, 0);
    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    Builder_.CreateStore(ZeroC, OffsetA);

    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = Builder_.CreateGEP(LhsPtrV, OffsetV);
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarTs[i]->getPointerTo());
      auto VarT = VarTs[i];
      auto InitC = initReduce(VarT, ReduceTypes[i]);
      Builder_.CreateStore(InitC, LhsPtrV, true /*volatile*/);
      auto SizeC = llvmValue(TheContext_, SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
    
    Builder_.CreateRetVoid();
  
    // device pointer
    InitPtrN = InitN + "_ptr";
    new GlobalVariable(
        TheModule, 
        InitT->getPointerTo(),
        false,
        GlobalValue::InternalLinkage,
        InitF, // has initializer, specified below
        InitPtrN,
        nullptr,
        GlobalValue::NotThreadLocal,
        1);
  }

  //----------------------------------------------------------------------------
  // create reduction

  return std::make_unique<CudaReduceInfo>(
      VarTs,
      ReduceTypes,
      InitF,
      InitPtrN,
      ApplyF,
      ApplyPtrN,
      FoldF,
      FoldPtrN);
}


}
