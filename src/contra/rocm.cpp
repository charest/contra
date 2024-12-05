#include "rocm.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
  
////////////////////////////////////////////////////////////////////////////////
// Rocm tasker
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
  Int64Type_ = llvmType<long long>(getContext());
  UInt32Type_ = llvmType<uint32_t>(getContext());
  UInt64Type_ = llvmType<uint64_t>(getContext());

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
  auto NewType = StructType::create( getContext(), members, "dim3" );
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
  auto NewType = StructType::create( getContext(), members, "dim3.reduced" );
  return NewType;
}

//==============================================================================
// Create the ihipstream type
//==============================================================================
StructType * ROCmTasker::createStreamType()
{
  auto NewType = StructType::create( getContext(), "ihipStream_t" );
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
  auto NewType = StructType::create( getContext(), members, "contra_rocm_partition_t" );
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
  auto NewType = StructType::create( getContext(), members, "contra_rocm_field_t" );
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
  auto NewType = StructType::create( getContext(), members, "contra_rocm_accessor_t" );
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
  auto StartPtrV = TheHelper_.offsetPointer(IntType_, OffsetsPtrV, IndexV);
  auto StartV  = getBuilder().CreateLoad(IntType_, StartPtrV);
  auto OneC = llvmValue<int_t>(getContext(), 1);
  auto IndexPlusOneV = getBuilder().CreateAdd(IndexV, OneC);
  auto EndPtrV = TheHelper_.offsetPointer(IntType_, OffsetsPtrV, IndexPlusOneV);
  auto EndV = getBuilder().CreateLoad(IntType_, EndPtrV);
  TheHelper_.insertValue(IndexA, StartV, 0);
  TheHelper_.insertValue(IndexA, EndV, 1);
  TheHelper_.insertValue(IndexA, OneC, 2);
}

//==============================================================================
// start runtime
//==============================================================================
void ROCmTasker::startRuntime(Module &TheModule)
{
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
    WrapperArgTs.emplace_back(ByteType_->getPointerTo(1));
    WrapperArgTs.emplace_back(ByteType_->getPointerTo(1));
    WrapperArgNs.emplace_back("indata");
    WrapperArgNs.emplace_back("outdata");
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
  // determine my index
 
  auto IndexV = getThreadID(TheModule);

  // cast and store
  auto IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, IntType_, "index");
  getBuilder().CreateStore(IndexV, IndexA);
  
  //----------------------------------------------------------------------------
  // If tid < total size

  auto TheFunction = getBuilder().GetInsertBlock()->getParent();
  BasicBlock *ThenBB = BasicBlock::Create(getContext(), "then", TheFunction);
  TaskI.MergeBlock = BasicBlock::Create(getContext(), "ifcont");

  auto IndexSizeA = WrapperArgAs[TaskArgNs.size()];
  auto IndexSizeV = TheHelper_.load(IndexSizeA);

  IndexV = TheHelper_.load(IndexA); 
  auto CondV = getBuilder().CreateICmpSLT(IndexV, IndexSizeV, "threadcond");
  getBuilder().CreateCondBr(CondV, ThenBB, TaskI.MergeBlock);
  
  // Emit then value.
  getBuilder().SetInsertPoint(ThenBB);

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
    TaskI.ResultBlockAlloca = WrapperArgAs.back();
    WrapperArgAs.pop_back();
    TaskI.ResultThreadAlloca = WrapperArgAs.back();
    WrapperArgAs.pop_back();
  }

  // Index size
  TaskI.IndexSizeAlloca = WrapperArgAs.back();
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
    AllocaInst* ResultA = nullptr;

    if (ResultV && !ResultV->getType()->isVoidTy()) {
      ResultA = TheHelper_.getAsAlloca(ResultV);
      auto ResultT = ResultA->getAllocatedType();
      auto ResultSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
      
      auto TidV = getThreadID(TheModule);
      auto PosV = getBuilder().CreateMul(TidV, ResultSizeV);
      auto IndataV = TheHelper_.load(TaskI.ResultThreadAlloca);
      auto OffsetV = TheHelper_.offsetPointer(ByteType_, IndataV, PosV);
      TheHelper_.memCopy(OffsetV, ResultA, ResultSizeV);

    }
  
    // finish If
    auto MergeBB = TaskI.MergeBlock;
    getBuilder().CreateBr(MergeBB);
    auto TheFunction = getBuilder().GetInsertBlock()->getParent();
    TheFunction->insert(TheFunction->end(), MergeBB);
    getBuilder().SetInsertPoint(MergeBB);
      
    if (ResultA) {  
      auto IndataV = TheHelper_.load(TaskI.ResultThreadAlloca);
      auto OutdataV = TheHelper_.load(TaskI.ResultBlockAlloca);
      auto IndexSizeV = TheHelper_.load(TaskI.IndexSizeAlloca);
      
      auto ResultT = ResultA->getAllocatedType();
      auto ResultSizeV = TheHelper_.getTypeSize<size_t>(ResultT);

      TheHelper_.callFunction(
        TheModule,
        "reduce",
        VoidType_,
        {
          IndataV,
          OutdataV,
          ResultSizeV,
          IndexSizeV
        });
    }
    

    getBuilder().CreateRetVoid();
  }
  //----------------------------------------------------------------------------
  // Single task
  else {

    // Have return value
    if (ResultV && !ResultV->getType()->isVoidTy()) {
      ResultV = TheHelper_.getAsValue(ResultV);
      getBuilder().CreateRet(ResultV);
    }
    // no return value
    else {
      getBuilder().CreateRetVoid();
    }

  }
  
  //----------------------------------------------------------------------------
  // Check if has printf
  
  TaskI.HasPrintf = TheModule.getFunction("print");


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
  getBuilder().CreateStore( getRangeSize(IndexSpaceA), IndexSpaceSizeA );
  ArgAs.emplace_back( IndexSpaceSizeA );
  
  //----------------------------------------------------------------------------
  // Serialize args
  
  auto TotArgs = ArgAs.size();

  std::vector<Type*> ArgTs;
  for (auto A : ArgAs) ArgTs.emplace_back( TheHelper_.getAllocatedType(A) );
  
  if (AbstractReduceOp) {
    ArgTs.emplace_back(ByteType_->getPointerTo(1));
    ArgTs.emplace_back(ByteType_->getPointerTo(1));
  }

  auto ArgsT = StructType::create( getContext(), ArgTs, "args_t" );
  auto ArgsA = TheHelper_.createEntryBlockAlloca(ArgsT, "args.alloca");

  for (unsigned i=0; i<TotArgs; ++i) {
    auto ArgV = TheHelper_.getAsValue(ArgAs[i]);
    TheHelper_.insertValue(ArgsA, ArgV, i);
  }

  auto ArgsSizeV = TheHelper_.getTypeSize(ArgsT, SizeType_);
    
  auto RangeA = TheHelper_.getAsAlloca(RangeV);
  
  //----------------------------------------------------------------------------
  // Prepare reduction function
  
  Value* ResultA = Constant::getNullValue(VoidPtrType_);
  Value* FoldF = Constant::getNullValue(VoidPtrType_);
  Value* IndataA = cast<Value>(Constant::getNullValue(VoidPtrType_->getPointerTo()));
  Value* OutdataA = cast<Value>(Constant::getNullValue(VoidPtrType_->getPointerTo()));
  Value* ResultSizeV = llvmValue<size_t>(getContext(), 0);

  if (AbstractReduceOp) {
    auto ReduceOp = dynamic_cast<const ROCmReduceInfo*>(AbstractReduceOp);
    
    auto ResultT = StructType::create( getContext(), "reduce" );
    ResultT->setBody( ReduceOp->getVarTypes() );
    ResultSizeV = TheHelper_.getTypeSize<size_t>(ResultT);
    
    const auto & FoldN = ReduceOp->getFoldName();
    auto FoldT = ReduceOp->getFoldType();
    FoldF = TheModule.getOrInsertFunction(FoldN, FoldT).getCallee();

    ResultA = TheHelper_.createEntryBlockAlloca(ResultT);

    IndataA = TheHelper_.getElementPointer(ArgsA, {0, static_cast<unsigned>(TotArgs)});
    OutdataA = TheHelper_.getElementPointer(ArgsA, {0, static_cast<unsigned>(TotArgs+1)});
  }
  
  //----------------------------------------------------------------------------
  // Launch function
      
  auto TaskStr = llvmString(getContext(), TheModule, TaskI.getName());
  TheHelper_.callFunction(
    TheModule,
    "contra_rocm_launch_kernel",
    VoidType_,
    {
      TaskStr,
      RangeA,
      ArgsA,
      ArgsSizeV,
      ResultA,
      ResultSizeV,
      IndataA,
      OutdataA,
      FoldF
    });

  
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
      llvmValue<bool>(getContext(), true)
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
bool ROCmTasker::isField(Value* Field) const
{
  auto FieldT = Field->getType();
  if (auto FieldA = dyn_cast<AllocaInst>(Field)) FieldT = FieldA->getAllocatedType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a rocm field
//==============================================================================
void ROCmTasker::createField(
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

bool ROCmTasker::isAccessor(Value* Accessor) const
{
  auto AccessorT = Accessor->getType();
  if (auto AccessorA = dyn_cast<AllocaInst>(Accessor)) 
    AccessorT = AccessorA->getAllocatedType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
std::pair<Value*, Value*> ROCmTasker::offsetAccessor(
    Module & TheModule,
    Value* AccessorV,
    Value* IndexV)
{
	auto IntPtrT = IntType_->getPointerTo();
  // get offsets
  std::vector<unsigned> Members;
  if (isa<AllocaInst>(AccessorV)) Members.emplace_back(0);
  Members.emplace_back(2); // partition
  Members.emplace_back(2); // offsets
  auto OffsetsPtrV = TheHelper_.getElementPointer(AccessorType_, AccessorV, Members);
  OffsetsPtrV = getBuilder().CreateLoad(IntPtrT, OffsetsPtrV);

  // get offsets[tid]
  auto TidV = getThreadID(TheModule);
  auto OffsetV = TheHelper_.offsetPointer(IntType_, OffsetsPtrV, TidV);
  OffsetV = getBuilder().CreateLoad(IntType_, OffsetV);

  // offsets[tid] + index
  if (IndexV) IndexV = TheHelper_.getAsValue(IndexV);
  else        IndexV = llvmValue<int_t>(getContext(), 0);
  auto PosV = getBuilder().CreateAdd(OffsetV, IndexV);

  // pos * data_size
  Members.clear();
  if (isa<AllocaInst>(AccessorV)) Members.emplace_back(0);
  Members.emplace_back(0);
  auto DataSizeV = TheHelper_.getElementPointer(AccessorType_, AccessorV, Members);
  DataSizeV = getBuilder().CreateLoad(IntType_, DataSizeV);
  PosV = getBuilder().CreateMul(PosV, DataSizeV);
  
  // data
  Members.clear();
  if (isa<AllocaInst>(AccessorV)) Members.emplace_back(0);
  Members.emplace_back(1);
  auto DataPtrV = TheHelper_.getElementPointer(AccessorType_, AccessorV, Members);
  DataPtrV = getBuilder().CreateLoad(IntPtrT, DataPtrV);
  
  // data + offset
  auto OffsetDataPtrV = TheHelper_.offsetPointer(ByteType_, DataPtrV, PosV);

  return {OffsetDataPtrV, DataSizeV};
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void ROCmTasker::storeAccessor(
    Module & TheModule,
    Value* ValueV,
    Value* AccessorV,
    Value* IndexV)
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
    Value* IndexV)
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

bool ROCmTasker::isPartition(Value* Part) const
{
  auto PartT = Part->getType();
  if (auto PartA = dyn_cast<AllocaInst>(Part)) PartT = PartA->getAllocatedType();
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
      "contra_rocm_partition_destroy",
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
  // create Fold
  Function * FoldF;
  std::string FoldPtrN;
  {
    std::string FoldN = "fold";

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

    auto BB = BasicBlock::Create(getContext(), "entry", FoldF);
    getBuilder().SetInsertPoint(BB);

    unsigned i=0;
    std::vector<AllocaInst*> ArgAs(ArgTs.size());
    for (auto &Arg : FoldF->args()) {
      ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
      getBuilder().CreateStore(&Arg, ArgAs[i]);
      ++i;
    }

    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    auto ZeroC = llvmValue(getContext(), SizeType_, 0);
    getBuilder().CreateStore(ZeroC, OffsetA);


    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      Value* RhsPtrV = TheHelper_.load(ArgAs[1]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = getBuilder().CreateGEP(ByteType_, LhsPtrV, OffsetV);
      RhsPtrV = getBuilder().CreateGEP(ByteType_, RhsPtrV, OffsetV);
      auto VarT = VarTs[i];
      auto VarPtrT = VarT->getPointerTo();
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarPtrT);
      RhsPtrV = TheHelper_.createBitCast(RhsPtrV, VarPtrT);
      auto LhsV = getBuilder().CreateLoad(VarT, LhsPtrV, true /*volatile*/);
      auto RhsV = getBuilder().CreateLoad(VarT, RhsPtrV, true /*volatile*/);
      auto ReduceV = foldReduce(TheModule, LhsV, RhsV, ReduceTypes[i]);
      getBuilder().CreateStore(ReduceV, LhsPtrV, true /*volatile*/);
      auto SizeC = llvmValue(getContext(), SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
        
    getBuilder().CreateRetVoid();
    
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

    auto BB = BasicBlock::Create(getContext(), "entry", ApplyF);
    getBuilder().SetInsertPoint(BB);

    unsigned i=0;
    std::vector<AllocaInst*> ArgAs(ArgTs.size());
    for (auto &Arg : ApplyF->args()) {
      ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
      getBuilder().CreateStore(&Arg, ArgAs[i]);
      ++i;
    }

    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    auto ZeroC = llvmValue(getContext(), SizeType_, 0);
    getBuilder().CreateStore(ZeroC, OffsetA);


    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      Value* RhsPtrV = TheHelper_.load(ArgAs[1]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = getBuilder().CreateGEP(ByteType_, LhsPtrV, OffsetV);
      RhsPtrV = getBuilder().CreateGEP(ByteType_, RhsPtrV, OffsetV);
      auto VarT = VarTs[i];
      auto VarPtrT = VarT->getPointerTo();
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarPtrT);
      RhsPtrV = TheHelper_.createBitCast(RhsPtrV, VarPtrT);
      auto LhsV = getBuilder().CreateLoad(VarT, LhsPtrV, true /*volatile*/);
      auto RhsV = getBuilder().CreateLoad(VarT, RhsPtrV, true /*volatile*/);
      auto ReduceV = applyReduce(TheModule, LhsV, RhsV, ReduceTypes[i]);
      getBuilder().CreateStore(ReduceV, LhsPtrV, true /*volatile*/);
      auto SizeC = llvmValue(getContext(), SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
        
    getBuilder().CreateRetVoid();
    
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
  
    auto BB = BasicBlock::Create(getContext(), "entry", InitF);
    getBuilder().SetInsertPoint(BB);
    
    unsigned i=0;
    std::vector<AllocaInst*> ArgAs(ArgTs.size());
    for (auto &Arg : InitF->args()) {
      ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
      getBuilder().CreateStore(&Arg, ArgAs[i]);
      ++i;
    }

    auto ZeroC = llvmValue(getContext(), SizeType_, 0);
    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    getBuilder().CreateStore(ZeroC, OffsetA);

    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = getBuilder().CreateGEP(ByteType_, LhsPtrV, OffsetV);
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarTs[i]->getPointerTo());
      auto VarT = VarTs[i];
      auto InitC = initReduce(VarT, ReduceTypes[i]);
      getBuilder().CreateStore(InitC, LhsPtrV, true /*volatile*/);
      auto SizeC = llvmValue(getContext(), SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
    
    getBuilder().CreateRetVoid();
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
Value* ROCmTasker::getThreadID(Module & TheModule) {
  // tid = threadIdx.x + blockIdx.x * blockDim.x;
  //Value* IndexV = TheHelper_.callFunction(
  //    TheModule,
  //    "__ockl_get_global_id",
  //    SizeType_,
  //    {llvmValue<uint>(getContext(), 0)} );
  

  // local_id
  auto TidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::amdgcn_workitem_id_x);
  Value* IndexV = getBuilder().CreateCall(TidF);

  // group_id
  auto BidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::amdgcn_workgroup_id_x);
  Value* BlockIdV = getBuilder().CreateCall(BidF);
  
  // group_size
  auto DispatchPtrF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::amdgcn_dispatch_ptr);
  Value* DispatchPtrV = getBuilder().CreateCall(DispatchPtrF);
  auto DispatchPtrGEP = TheHelper_.getElementPointer(ByteType_, DispatchPtrV, 4); 
  auto Int16T = llvmType<uint16_t>(getContext());
  auto AddrSpace = DispatchPtrGEP->getType()->getPointerAddressSpace();
  auto Int16PtrT = Int16T->getPointerTo( AddrSpace );
  auto BlockDimPtrV = TheHelper_.createBitCast(DispatchPtrGEP, Int16PtrT);
  Value* GroupSizeV = getBuilder().CreateLoad(Int16T, BlockDimPtrV);
  auto IdT = BlockIdV->getType();
  GroupSizeV = TheHelper_.createCast(GroupSizeV, IdT);

  // grid_size
  DispatchPtrGEP = TheHelper_.getElementPointer(ByteType_, DispatchPtrV, 12);
  auto Int32T = llvmType<uint32_t>(getContext());
  auto Int32PtrT = Int32T->getPointerTo( AddrSpace );
  auto GridSizePtrV = TheHelper_.createBitCast(DispatchPtrGEP, Int32PtrT);
  Value* GridSizeV = getBuilder().CreateLoad(Int32T, GridSizePtrV);
  GridSizeV = TheHelper_.createCast(GridSizeV, IdT);

  // r = grid_size - group_id * group_size
  Value* TmpV = getBuilder().CreateMul(BlockIdV, GroupSizeV);
  TmpV = getBuilder().CreateSub(GridSizeV, TmpV);

  // local_size = (r < group_size) ? r : group_size;
  auto CondV = getBuilder().CreateICmpULT(TmpV, GroupSizeV, "threadcond");
  auto BlockDimV = getBuilder().CreateSelect(CondV, TmpV, GroupSizeV);
  
  // local_id + group_id*local_size
  TmpV = getBuilder().CreateMul(BlockIdV, BlockDimV);
  IndexV = getBuilder().CreateAdd(IndexV, TmpV);
  IndexV = TheHelper_.createCast(IndexV, IntType_);

  return IndexV;
}

} // namespace
