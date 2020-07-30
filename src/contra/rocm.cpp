#include "rocm.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"
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
// Create an index space from a partition
//==============================================================================
void ROCmTasker::createIndexSpaceFromPartition(
    Value* IndexV,
    AllocaInst* PartA,
    AllocaInst* IndexA)
{
  auto OffsetsPtrV = TheHelper_.extractValue(PartA, 2);
  auto StartPtrV = TheHelper_.offsetPointer(OffsetsPtrV, IndexV);
  auto StartV  = TheHelper_.load(StartPtrV);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  auto IndexPlusOneV = Builder_.CreateAdd(IndexV, OneC);
  auto EndPtrV = TheHelper_.offsetPointer(OffsetsPtrV, IndexPlusOneV);
  auto EndV = TheHelper_.load(EndPtrV);
  TheHelper_.insertValue(IndexA, StartV, 0);
  TheHelper_.insertValue(IndexA, EndV, 1);
  TheHelper_.insertValue(IndexA, OneC, 2);
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
  WrapperF->setVisibility(GlobalValue::ProtectedVisibility);
  WrapperF->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  // annotate as kernel
  WrapperF->setCallingConv(CallingConv::AMDGPU_KERNEL);

  //WrapperF->addFnAttr("amdgpu-flat-work-group-size", "1,256");
  WrapperF->addFnAttr("amdgpu-implicitarg-num-bytes", "56");
  //WrapperF->addFnAttr("correctly-rounded-divide-sqrt-fp-math", "false");
  //WrapperF->addFnAttr("denormal-fp-math-f32", "preserve-sign,preserve-sign");
  //WrapperF->addFnAttr("disable-tail-calls", "false");
  //WrapperF->addFnAttr("frame-pointer", "none");
  //WrapperF->addFnAttr("less-precise-fpmad", "false");
  //WrapperF->addFnAttr("min-legal-vector-width", "0");
  //WrapperF->addFnAttr("no-infs-fp-math", "false");
  //WrapperF->addFnAttr("no-jump-tables", "false");
  //WrapperF->addFnAttr("no-nans-fp-math", "false");
  //WrapperF->addFnAttr("no-signed-zeros-fp-math", "false");
  //WrapperF->addFnAttr("no-trapping-math", "false");
  //WrapperF->addFnAttr("stack-protector-buffer-size", "8");
  //WrapperF->addFnAttr("target-cpu", "gfx900");
  //WrapperF->addFnAttr("target-features", "+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime");
  //WrapperF->addFnAttr("uniform-work-group-size", "true");
  //WrapperF->addFnAttr("unsafe-fp-math", "false");
  //WrapperF->addFnAttr("use-soft-float", "false");
  
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
 
  auto IndexV = getThreadID(TheModule);

  // cast and store
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
  
  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    if (isRange(TaskArgTs[i])) {
      auto ArgN = TaskArgNs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      createIndexSpaceFromPartition(IndexV, WrapperArgAs[i], ArgA);
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
      
      auto TidV = getThreadID(TheModule);
      auto PosV = Builder_.CreateMul(TidV, DataSizeV);
      auto IndataV = TheHelper_.load(IndataA);
      auto OffsetV = TheHelper_.offsetPointer(IndataV, PosV);
      TheHelper_.memCopy(OffsetV, ResultA, DataSizeV);
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
      auto ArrayT = librt::DopeVector::DopeVectorType;
      auto DevPtrA = TheHelper_.createEntryBlockAlloca(ArrayT);
      TheHelper_.callFunction(
          TheModule,
          "contra_rocm_array2dev",
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
  
  auto TotArgs = ArgAs.size();

  std::vector<Type*> ArgTs;
  for (auto A : ArgAs) ArgTs.emplace_back( TheHelper_.getAllocatedType(A) );

  auto ArgsT = StructType::create( TheContext_, ArgTs, "args_t" );
  auto ArgsA = TheHelper_.createEntryBlockAlloca(ArgsT, "args.alloca");

  for (unsigned i=0; i<TotArgs; ++i) {
    auto ArgV = TheHelper_.getAsValue(ArgAs[i]);
    TheHelper_.insertValue(ArgsA, ArgV, i);
  }

  auto ArgsSizeV = TheHelper_.getTypeSize(ArgsT, SizeType_);
    
  auto RangeA = TheHelper_.getAsAlloca(RangeV);
  
  //----------------------------------------------------------------------------
  // Register function
  
  
  auto TaskStr = llvmString(TheContext_, TheModule, TaskI.getName());
  TheHelper_.callFunction(
    TheModule,
    "contra_rocm_launch_kernel",
    VoidType_,
    {TaskStr, RangeA, ArgsA, ArgsSizeV} );

  //----------------------------------------------------------------------------
  // Call function with reduction
  AllocaInst* ResultA = nullptr;
  if (ResultT && AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const ROCmReduceInfo*>(AbstractReduceOp);
    
    const auto & ApplyN = ReduceOp->getApplyName();
    auto ApplyT = ReduceOp->getApplyType();
    auto ApplyF = TheModule.getOrInsertFunction(ApplyN, ApplyT).getCallee();

    ResultA = TheHelper_.createEntryBlockAlloca(ResultT);
    auto OutDataV = TheHelper_.createBitCast(ResultA, VoidPtrType_);
    auto DataSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
    std::vector<Value*> ArgVs = {
      TaskStr,
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
std::pair<Value*, Value*> ROCmTasker::offsetAccessor(
    Module & TheModule,
    Value* AccessorV,
    Value* IndexV) const
{
  // get offsets
  std::vector<unsigned> Members;
  if (isa<AllocaInst>(AccessorV)) Members.emplace_back(0);
  Members.emplace_back(2); // partition
  Members.emplace_back(2); // offsets
  auto OffsetsPtrV = TheHelper_.getElementPointer(AccessorV, Members);
  OffsetsPtrV = TheHelper_.load(OffsetsPtrV);

  // get offsets[tid]
  auto TidV = getThreadID(TheModule);
  auto OffsetV = TheHelper_.offsetPointer(OffsetsPtrV, TidV);
  OffsetV = TheHelper_.load(OffsetV);

  // offsets[tid] + index
  if (IndexV) IndexV = TheHelper_.getAsValue(IndexV);
  else        IndexV = llvmValue<int_t>(TheContext_, 0);
  auto PosV = Builder_.CreateAdd(OffsetV, IndexV);

  // pos * data_size
  Members.clear();
  if (isa<AllocaInst>(AccessorV)) Members.emplace_back(0);
  Members.emplace_back(0);
  auto DataSizeV = TheHelper_.getElementPointer(AccessorV, Members);
  DataSizeV = TheHelper_.load(DataSizeV);
  PosV = Builder_.CreateMul(PosV, DataSizeV);
  
  // data
  Members.clear();
  if (isa<AllocaInst>(AccessorV)) Members.emplace_back(0);
  Members.emplace_back(1);
  auto DataPtrV = TheHelper_.getElementPointer(AccessorV, Members);
  DataPtrV = TheHelper_.load(DataPtrV);
  
  // data + offset
  auto OffsetDataPtrV = TheHelper_.offsetPointer(DataPtrV, PosV);

  return {OffsetDataPtrV, DataSizeV};
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
  auto res = offsetAccessor(TheModule, AccessorV, IndexV);
  auto OffsetDataPtrV = res.first;
  auto DataSizeV = res.second;

  auto ValueA = TheHelper_.getAsAlloca(ValueV);
  ValueV = TheHelper_.createBitCast(ValueA, VoidPtrType_);

  TheHelper_.memCopy(OffsetDataPtrV, ValueV, DataSizeV);
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
  // get pointer to data
  auto res = offsetAccessor(TheModule, AccessorV, IndexV);
  auto OffsetDataPtrV = res.first;
  auto DataSizeV = res.second;

  // create result alloca
  Value* ValueA = TheHelper_.createEntryBlockAlloca(ValueT);
  auto ValuePtr = TheHelper_.createBitCast(ValueA, VoidPtrType_);

  // memcopy
  TheHelper_.memCopy(ValuePtr, OffsetDataPtrV, DataSizeV);
  
  return ValueA;
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
    std::string ApplyN = "apply";

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
    
  }

  //----------------------------------------------------------------------------
  // create fold
  Function * FoldF;
  std::string FoldPtrN;
  {
    std::string FoldN = "fold";

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
    
  }


  //----------------------------------------------------------------------------
  // create init
  Function * InitF;
  std::string InitPtrN;
  {

    std::string InitN = "init";
 
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


//==============================================================================
// Get the thread id
//==============================================================================
Value* ROCmTasker::getThreadID(Module & TheModule) const {
  // tid = threadIdx.x + blockIdx.x * blockDim.x;
  //Value* IndexV = TheHelper_.callFunction(
  //    TheModule,
  //    "__ockl_get_global_id",
  //    SizeType_,
  //    {llvmValue<uint>(TheContext_, 0)} );
  

  // local_id
  auto TidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::amdgcn_workitem_id_x);
  Value* IndexV = Builder_.CreateCall(TidF);

  // group_id
  auto BidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::amdgcn_workgroup_id_x);
  Value* BlockIdV = Builder_.CreateCall(BidF);
  
  // group_size
  auto DispatchPtrF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::amdgcn_dispatch_ptr);
  Value* DispatchPtrV = Builder_.CreateCall(DispatchPtrF);
  auto DispatchPtrGEP = TheHelper_.getElementPointer(DispatchPtrV, 4); 
  auto Int16T = llvmType<uint16_t>(TheContext_);
  auto AddrSpace = DispatchPtrGEP->getType()->getPointerAddressSpace();
  auto Int16PtrT = Int16T->getPointerTo( AddrSpace );
  auto BlockDimPtrV = TheHelper_.createBitCast(DispatchPtrGEP, Int16PtrT);
  Value* GroupSizeV = TheHelper_.load(BlockDimPtrV);
  auto IdT = BlockIdV->getType();
  GroupSizeV = TheHelper_.createCast(GroupSizeV, IdT);

  // grid_size
  DispatchPtrGEP = TheHelper_.getElementPointer(DispatchPtrV, 12);
  auto Int32T = llvmType<uint32_t>(TheContext_);
  auto Int32PtrT = Int32T->getPointerTo( AddrSpace );
  auto GridSizePtrV = TheHelper_.createBitCast(DispatchPtrGEP, Int32PtrT);
  Value* GridSizeV = TheHelper_.load(GridSizePtrV);
  GridSizeV = TheHelper_.createCast(GridSizeV, IdT);

  // r = grid_size - group_id * group_size
  Value* TmpV = Builder_.CreateMul(BlockIdV, GroupSizeV);
  TmpV = Builder_.CreateSub(GridSizeV, TmpV);

  // local_size = (r < group_size) ? r : group_size;
  auto CondV = Builder_.CreateICmpULT(TmpV, GroupSizeV, "threadcond");
  auto BlockDimV = Builder_.CreateSelect(CondV, TmpV, GroupSizeV);
  
  // local_id + group_id*local_size
  TmpV = Builder_.CreateMul(BlockIdV, BlockDimV);
  IndexV = Builder_.CreateAdd(IndexV, TmpV);
  IndexV = TheHelper_.createCast(IndexV, IntType_);

  return IndexV;
}

} // namespace
