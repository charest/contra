#include "rocm.hpp"

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
ROCmTasker::ROCmTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  Int64Type_ = llvmType<long long>(TheContext_);
  UInt32Type_ = llvmType<uint32_t>(TheContext_);
  UInt64Type_ = llvmType<uint64_t>(TheContext_);

  Dim3Type_ = createDim3Type();
  ReducedDim3Type_ = createReducedDim3Type();
  StreamType_ = createStreamType();

  IndexSpaceType_ = DefaultIndexSpaceType_;
  IndexPartitionType_ = createIndexPartitionType();
  PartitionInfoType_ = VoidPtrType_->getPointerTo();
  TaskInfoType_ = VoidPtrType_->getPointerTo();
  FieldType_ = createFieldType();
  AccessorType_ = createAccessorType();
}

//==============================================================================
// Create the dim3 type
//==============================================================================
StructType * ROCmTasker::createDim3Type()
{
  std::vector<Type*> members = {
    UInt32Type_,
    UInt32Type_,
    UInt32Type_};
  auto NewType = StructType::create( TheContext_, members, "dim3" );
  return NewType;
}

//==============================================================================
// Create the dim3 type
//==============================================================================
StructType * ROCmTasker::createReducedDim3Type()
{
  std::vector<Type*> members = {
    UInt64Type_,
    UInt32Type_};
  auto NewType = StructType::create( TheContext_, members, "dim3.reduced" );
  return NewType;
}

//==============================================================================
// Create the ihipstream type
//==============================================================================
StructType * ROCmTasker::createStreamType()
{
  auto NewType = StructType::create( TheContext_, "ihipStream_t" );
  return NewType;
}


//==============================================================================
// Create the partition data type
//==============================================================================
StructType * ROCmTasker::createIndexPartitionType()
{
  auto IntPtrT = IntType_->getPointerTo();
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    IntPtrT,
    IntPtrT};
  auto NewType = StructType::create( TheContext_, members, "contra_rocm_partition_t" );
  return NewType;
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * ROCmTasker::createFieldType()
{
  std::vector<Type*> members = {
    IntType_,
    IntType_,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo()};
  auto NewType = StructType::create( TheContext_, members, "contra_rocm_field_t" );
  return NewType;
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * ROCmTasker::createAccessorType()
{
  std::vector<Type*> members = {
    IntType_,
    VoidPtrType_,
    IndexPartitionType_,
    FieldType_->getPointerTo() };
  auto NewType = StructType::create( TheContext_, members, "contra_rocm_accessor_t" );
  return NewType;
}


//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* ROCmTasker::createPartitionInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(PartitionInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_rocm_partition_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void ROCmTasker::destroyPartitionInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_rocm_partition_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// Create partitioninfo
//==============================================================================
AllocaInst* ROCmTasker::createTaskInfo(Module & TheModule)
{
  auto Alloca = TheHelper_.createEntryBlockAlloca(TaskInfoType_);
  TheHelper_.callFunction(
      TheModule,
      "contra_rocm_task_info_create",
      VoidType_,
      {Alloca});
  return Alloca;
}

//==============================================================================
// destroy partition info
//==============================================================================
void ROCmTasker::destroyTaskInfo(Module & TheModule, AllocaInst* PartA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_rocm_task_info_destroy",
      VoidType_,
      {PartA});
}

//==============================================================================
// start runtime
//==============================================================================
void ROCmTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_rocm_startup",
      VoidType_);

  launch(TheModule, *TopLevelTask_);
}
//==============================================================================
// Default Preamble
//==============================================================================
ROCmTasker::PreambleResult ROCmTasker::taskPreamble(
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
ROCmTasker::PreambleResult ROCmTasker::taskPreamble(
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
  
  //WrapperArgTs.emplace_back(IntType_);
  //WrapperArgNs.emplace_back("size");

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
  WrapperF->setVisibility(GlobalValue::ProtectedVisibility);
  WrapperF->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  // annotate as kernel
  WrapperF->setCallingConv(CallingConv::AMDGPU_KERNEL);

  std::vector< std::pair<std::string,std::string> > Annotes = {
    //{"rocdl.hsaco", "HSACO"},
    {"amdgpu-flat-work-group-size", "1,256"},
    {"amdgpu-implicitarg-num-bytes", "56"},
    {"correctly-rounded-divide-sqrt-fp-math", "false"},
    {"denormal-fp-math-f32", "preserve-sign,preserve-sign"},
    {"disable-tail-calls", "false"},
    {"frame-pointer", "none"},
    {"less-precise-fpmad", "false"},
    {"min-legal-vector-width", "0"},
    {"no-infs-fp-math", "false"},
    {"no-jump-tables", "false"},
    {"no-nans-fp-math", "false"},
    {"no-signed-zeros-fp-math", "false"},
    {"no-trapping-math", "false"},
    {"stack-protector-buffer-size", "8"},
    {"target-cpu", "gfx900"},
    {"target-features", "+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime"},
    {"uniform-work-group-size", "true"},
    {"unsafe-fp-math", "false"},
    {"use-soft-float", "false"} };

  for ( const auto & Ann : Annotes ) {
    WrapperF->addFnAttr(Ann.first, Ann.second);
  }
  
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
  // __ockl_get_global_id
  //auto TidF = Intrinsic::getDeclaration(
  //    &TheModule,
  //    Intrinsic::nvvm_read_ptx_sreg_tid_x);
  //Value* IndexV = Builder_.CreateCall(TidF);
  //auto BidF = Intrinsic::getDeclaration(
  //    &TheModule,
  //    Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  //Value* BlockIdV = Builder_.CreateCall(BidF);
  //auto BdimF = Intrinsic::getDeclaration(
  //    &TheModule,
  //    Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  //Value* BlockDimV = Builder_.CreateCall(BdimF);

  //Value* TmpV = Builder_.CreateMul(BlockIdV, BlockDimV);
  //IndexV = Builder_.CreateAdd(IndexV, TmpV);

  // cast and store
  //IndexV = TheHelper_.createCast(IndexV, IntType_);
  Value* IndexV = llvmValue(TheContext_, IntType_, 0);
  auto IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, IntType_, "index");
  Builder_.CreateStore(IndexV, IndexA);
#if 0
  
  //----------------------------------------------------------------------------
  // If tid < total size

  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  BasicBlock *ThenBB = BasicBlock::Create(TheContext_, "then", TheFunction);
  TaskI.MergeBlock = BasicBlock::Create(TheContext_, "ifcont");

#if 0
  auto IndexSizeA = WrapperArgAs[TaskArgNs.size()];
  auto IndexSizeV = TheHelper_.load(IndexSizeA);
#else
  auto IndexSizeV = llvmValue(TheContext_, IntType_, 1);
#endif

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
      "contra_rocm_index_space_create_from_partition",
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
#endif

  if (ResultT) {
    TaskI.ResultAlloca = WrapperArgAs.back();
    WrapperArgAs.pop_back();
  }

  // Index size
  //WrapperArgAs.pop_back();
  
  return {WrapperF, WrapperArgAs, IndexA};
}

//==============================================================================
// Create the function wrapper
//==============================================================================
void ROCmTasker::taskPostamble(
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
          "contra_rocm_set_reduction_value",
          VoidType_,
          {IndataA, ResultA, DataSizeV});
    }
  
    // finish If
    //auto MergeBB = TaskI.MergeBlock;
    //Builder_.CreateBr(MergeBB);
    //auto TheFunction = Builder_.GetInsertBlock()->getParent();
    //TheFunction->getBasicBlockList().push_back(MergeBB);
    //Builder_.SetInsertPoint(MergeBB);

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
Value* ROCmTasker::launch(
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
          "contra_rocm_partition2dev",
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
      auto DevPtr = TheHelper_.callFunction(
          TheModule,
          "contra_rocm_array2dev",
          librt::DopeVector::DopeVectorType,
          {ArrayA});
      auto DevPtrA = TheHelper_.getAsAlloca(DevPtr);
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
            "contra_rocm_field2dev",
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
    auto ReduceOp = dynamic_cast<const ROCmReduceInfo*>(AbstractReduceOp);
    
    ResultT = StructType::create( TheContext_, "reduce" );
    ResultT->setBody( ReduceOp->getVarTypes() );
    IndataA = TheHelper_.createEntryBlockAlloca(VoidPtrType_);
    auto DataSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
    TheHelper_.callFunction(
        TheModule,
        "contra_rocm_prepare_reduction",
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
  // Register function
  
#if 0
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  //TaskF->setCallingConv(CallingConv::AMDGPU_KERNEL);
  auto TaskPtr = TheHelper_.createBitCast(TheFunction, VoidPtrType_);
  auto TaskStr = llvmString(TheContext_, TheModule, TaskI.getName());

  // declare dso_local i32 @__hipRegisterFunction(i8**, i8*, i8*, i8*, i32, i8*, i8*, i8*, i8*, i32*) local_unnamed_addr
  
  std::vector<Type*> LaunchArgsT = {
    VoidPtrType_,
    UInt64Type_, UInt32Type_,
    UInt64Type_, UInt32Type_,
    VoidPtrPtrT,
    SizeType_,
    StreamPtrT};
  auto LaunchT = FunctionType::get(Int32Type_, LaunchArgsT, false);
  auto LaunchF = Function::Create(
      LaunchT,
      Function::ExternalLinkage,
      "hipLaunchKernel",
      &TheModule);
    
  //TheModule.print(outs(), nullptr);
  std::vector<Value*> LaunchArgsV = {
    TaskPtr,
    NumBlocks1V, NumBlocks2V,
    DimBlocks1V, DimBlocks2V,
    ParamsV,
    SharedMemBytesV,
    StreamV};
  Builder_.CreateCall(LaunchF, LaunchArgsV);
  
  //----------------------------------------------------------------------------
  // Call function

  auto ZeroC = llvmValue(TheContext_, UInt32Type_, 0);
  auto OneC = llvmValue(TheContext_, UInt32Type_, 1);

  auto NumBlocksA = TheHelper_.createEntryBlockAlloca(Dim3Type_);
  TheHelper_.insertValue(NumBlocksA, OneC, 0);
  TheHelper_.insertValue(NumBlocksA, ZeroC, 1);
  TheHelper_.insertValue(NumBlocksA, ZeroC, 2);

  auto DimBlocksA = TheHelper_.createEntryBlockAlloca(Dim3Type_);
  TheHelper_.insertValue(DimBlocksA, OneC, 0);
  TheHelper_.insertValue(DimBlocksA, ZeroC, 1);
  TheHelper_.insertValue(DimBlocksA, ZeroC, 2);

  auto ReducedDim3PtrT = ReducedDim3Type_->getPointerTo();
  Value* NumBlocksPtr = TheHelper_.createBitCast(NumBlocksA, ReducedDim3PtrT);
  auto NumBlocks1V = TheHelper_.getElementPointer(NumBlocksPtr, {0, 0});
  auto NumBlocks2V = TheHelper_.getElementPointer(NumBlocksPtr, {0, 1});
  NumBlocks1V = TheHelper_.load(NumBlocks1V);
  NumBlocks2V = TheHelper_.load(NumBlocks2V);
  
  Value* DimBlocksPtr = TheHelper_.createBitCast(DimBlocksA, ReducedDim3PtrT);
  auto DimBlocks1V = TheHelper_.getElementPointer(DimBlocksPtr, {0, 0});
  auto DimBlocks2V = TheHelper_.getElementPointer(DimBlocksPtr, {0, 1});
  DimBlocks1V = TheHelper_.load(DimBlocks1V);
  DimBlocks2V = TheHelper_.load(DimBlocks2V);

  auto VoidPtrPtrT = VoidPtrType_->getPointerTo();
  auto ParamsV = Constant::getNullValue(VoidPtrPtrT);
  auto SharedMemBytesV = llvmValue(TheContext_, SizeType_, 0);

  auto StreamPtrT = StreamType_->getPointerTo();
  auto StreamV = Constant::getNullValue(StreamPtrT);

  std::vector<Type*> LaunchArgsT = {
    VoidPtrType_,
    UInt64Type_, UInt32Type_,
    UInt64Type_, UInt32Type_,
    VoidPtrPtrT,
    SizeType_,
    StreamPtrT};
  auto LaunchT = FunctionType::get(Int32Type_, LaunchArgsT, false);
  auto LaunchF = Function::Create(
      LaunchT,
      Function::ExternalLinkage,
      "hipLaunchKernel",
      &TheModule);
    
  //TheModule.print(outs(), nullptr);
  std::vector<Value*> LaunchArgsV = {
    TaskPtr,
    NumBlocks1V, NumBlocks2V,
    DimBlocks1V, DimBlocks2V,
    ParamsV,
    SharedMemBytesV,
    StreamV};
  //Builder_.CreateCall(LaunchF, LaunchArgsV);
#endif
  
  auto TaskStr = llvmString(TheContext_, TheModule, TaskI.getName());
  TheHelper_.callFunction(
    TheModule,
    "contra_rocm_launch_kernel",
    VoidType_,
    {TaskStr, RangeA/*, ArgsA*/});

  //----------------------------------------------------------------------------
  // Call function with reduction
  AllocaInst* ResultA = nullptr;
#if 0
  if (ResultT && AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const ROCmReduceInfo*>(AbstractReduceOp);

    auto InitStr = llvmString(TheContext_, TheModule, ReduceOp->getInitPtrName());
    auto ApplyStr = llvmString(TheContext_, TheModule, ReduceOp->getApplyPtrName());
    auto FoldStr = llvmString(TheContext_, TheModule, ReduceOp->getFoldPtrName());

    const auto & ApplyN = ReduceOp->getApplyName();
    auto ApplyT = ReduceOp->getApplyType();
    auto ApplyF = TheModule.getOrInsertFunction(ApplyN, ApplyT).getCallee();

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
      ApplyF
    };
    TheHelper_.callFunction(
        TheModule,
        "contra_rocm_launch_reduction",
        VoidType_,
        ArgVs);
  }
#endif
  
  //----------------------------------------------------------------------------
  // cleanup

  for (auto AllocA : ToFree) {
    if (isPartition(AllocA)) {
      TheHelper_.callFunction(
          TheModule,
          "contra_rocm_partition_free_and_deregister",
          VoidType_,
          {AllocA, TaskInfoA});
    }
    else if (isAccessor(AllocA)) {
      TheHelper_.callFunction(
          TheModule,
          "contra_rocm_accessor_free_temp",
          VoidType_,
          {AllocA, TaskInfoA});
    }
    else if (librt::DopeVector::isDopeVector(AllocA)) {
      TheHelper_.callFunction(
          TheModule,
          "contra_rocm_array_free",
          VoidType_,
          {AllocA});
    }
    else {
      TheHelper_.callFunction(
          TheModule,
          "contra_rocm_free",
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
AllocaInst* ROCmTasker::createPartition(
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
        "contra_rocm_partition_from_index_space",
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
        "contra_rocm_partition_from_array",
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
        "contra_rocm_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  
  return IndexPartA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* ROCmTasker::createPartition(
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
        "contra_rocm_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
}

//==============================================================================
// Is this a field type
//==============================================================================
bool ROCmTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a legion field
//==============================================================================
void ROCmTasker::createField(
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
      "contra_rocm_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void ROCmTasker::destroyField(Module &TheModule, Value* FieldA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_rocm_field_destroy",
      VoidType_,
      {FieldA});
}

//==============================================================================
// Is this an accessor type
//==============================================================================
bool ROCmTasker::isAccessor(Type* AccessorT) const
{ return (AccessorT == AccessorType_); }

bool ROCmTasker::isAccessor(Value* AccessorA) const
{
  auto AccessorT = AccessorA->getType();
  if (isa<AllocaInst>(AccessorA)) AccessorT = AccessorT->getPointerElementType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void ROCmTasker::storeAccessor(
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
      "contra_rocm_accessor_write",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* ROCmTasker::loadAccessor(
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
      "contra_rocm_accessor_read",
      VoidType_,
      FunArgVs);

  return TheHelper_.load(ValueA);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void ROCmTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  //TheHelper_.callFunction(
  //    TheModule,
  //    "contra_rocm_accessor_destroy",
  //    VoidType_,
  //    {AccessorA});
}

//==============================================================================
// Is this an range type
//==============================================================================
bool ROCmTasker::isPartition(Type* PartT) const
{ return (PartT == IndexPartitionType_); }

bool ROCmTasker::isPartition(Value* PartA) const
{
  auto PartT = PartA->getType();
  if (isa<AllocaInst>(PartA)) PartT = PartT->getPointerElementType();
  return isPartition(PartT);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void ROCmTasker::destroyPartition(
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
std::unique_ptr<AbstractReduceInfo> ROCmTasker::createReductionOp(
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
  // create apply
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
  // create fold
  Function * FoldF;
  std::string FoldPtrN;
  {
    std::string FoldN = ReductionN + "fold";

    std::vector<Type*> ArgTs = {
      VoidPtrType_,
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
      auto VarT = VarTs[i];
      auto VarPtrT = VarT->getPointerTo();
      // res += lhs + rhs
      // 1. lhs + rhs
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      Value* RhsPtrV = TheHelper_.load(ArgAs[1]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = Builder_.CreateGEP(LhsPtrV, OffsetV);
      RhsPtrV = Builder_.CreateGEP(RhsPtrV, OffsetV);
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarPtrT);
      RhsPtrV = TheHelper_.createBitCast(RhsPtrV, VarPtrT);
      auto LhsV = Builder_.CreateLoad(VarT, LhsPtrV, true /*volatile*/);
      auto RhsV = Builder_.CreateLoad(VarT, RhsPtrV, true /*volatile*/);
      auto ReduceV = foldReduce(TheModule, LhsV, RhsV, ReduceTypes[i]);
      // 2. res + previous
      Value* ResPtrV = TheHelper_.load(ArgAs[2]);
      OffsetV = TheHelper_.load(OffsetA);
      ResPtrV = Builder_.CreateGEP(ResPtrV, OffsetV);
      ResPtrV = TheHelper_.createBitCast(ResPtrV, VarPtrT);
      auto ResV = Builder_.CreateLoad(VarT, ResPtrV, true /*volatile*/);
      ReduceV = applyReduce(TheModule, ResV, ReduceV, ReduceTypes[i]);
      // 3. store
      Builder_.CreateStore(ReduceV, ResPtrV, true /*volatile*/);
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

  return std::make_unique<ROCmReduceInfo>(
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
