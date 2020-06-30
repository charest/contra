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
  FieldType_ = createFieldType();
  AccessorType_ = createAccessorType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * CudaTasker::createFieldType()
{
  std::vector<Type*> members = {
    IntType_,
    VoidPtrType_,
    IndexSpaceType_->getPointerTo() };
  auto NewType = StructType::create( TheContext_, members, "contra_serial_field_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * CudaTasker::createAccessorType()
{
  std::vector<Type*> members = {
    BoolType_,
    IntType_,
    VoidPtrType_};
  auto NewType = StructType::create( TheContext_, members, "contra_serial_accessor_t" );
  return NewType;
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
    IntPtrT->getPointerTo(),
    IndexSpaceType_->getPointerTo()};
  auto NewType = StructType::create( TheContext_, members, "contra_serial_partition_t" );
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
      "contra_serial_partition_info_create",
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
      "contra_serial_partition_info_destroy",
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
// Create the function wrapper
//==============================================================================
CudaTasker::PreambleResult CudaTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    llvm::Type* ResultT)
{

  //----------------------------------------------------------------------------
  // Create function header
  std::vector<std::string> WrapperArgNs = TaskArgNs;
    
  std::vector<Type*> WrapperArgTs;
  for (auto ArgT : TaskArgTs) {
    if (isRange(ArgT)) ArgT = IndexPartitionType_;
    WrapperArgTs.emplace_back(ArgT);
  }

  if (!ResultT) ResultT = VoidType_;

  auto WrapperT = FunctionType::get(ResultT, WrapperArgTs, false);
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
  // partition any ranges
  std::vector<Type*> GetRangeArgTs = {
    IntType_,
    IndexPartitionType_->getPointerTo(),
    IndexSpaceType_->getPointerTo()
  };

#if 0
  auto GetRangeF = TheHelper_.createFunction(
      TheModule,
      "contra_serial_index_space_create_from_partition",
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
  
  //----------------------------------------------------------------------------
  // determine my index
  
  auto TidF = Intrinsic::getDeclaration(
      &TheModule,
      Intrinsic::nvvm_read_ptx_sreg_tid_x);
  Value* IndexV = Builder_.CreateCall(TidF);
  IndexV = TheHelper_.createCast(IndexV, IntType_);
  auto IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, IntType_, "index");
  Builder_.CreateStore(IndexV, IndexA);

  return {WrapperF, WrapperArgAs, IndexA};
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
    auto ReduceOp = dynamic_cast<const CudaReduceInfo*>(AbstractReduceOp);
    
    auto ResultT = StructType::create( TheContext_, "reduce" );
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
  // Setup args

  std::vector<Value*> ArgVs;
  for (auto ArgA : ArgAs) {
    if (librt::DopeVector::isDopeVector(ArgA)) {
      auto ArrayA = TheHelper_.getAsAlloca(ArgA);
      auto DevPtr = TheHelper_.callFunction(
          TheModule,
          "contra_cuda_array2dev",
          VoidPtrType_,
          {ArrayA});
      ArgVs.emplace_back( TheHelper_.getAsAlloca(DevPtr) );
      ArgVs.emplace_back( TheHelper_.getElementPointer(ArrayA, 1) );
      ArgVs.emplace_back( ArgVs.back() );
      ArgVs.emplace_back( TheHelper_.getElementPointer(ArrayA, 3) );
    }
    else {
      ArgVs.emplace_back( TheHelper_.getAsAlloca(ArgA) );
    }
  }

  auto NumExpandedArgs = ArgVs.size();
  auto ExpandedArgsT = ArrayType::get(VoidPtrType_, NumExpandedArgs);
  auto ExpandedArgsA = TheHelper_.createEntryBlockAlloca(ExpandedArgsT);

  for (unsigned i=0; i<NumExpandedArgs; ++i) {
    auto ArgV = TheHelper_.createBitCast(ArgVs[i], VoidPtrType_);
    TheHelper_.insertValue(ExpandedArgsA, ArgV, i);
  }
    
  auto RangeA = TheHelper_.getAsAlloca(RangeV);
  

  //----------------------------------------------------------------------------
  // Call function

  //------------------------------------
  // Call function with reduction
  if (ResultA && AbstractReduceOp) {
    auto ResultT = ResultA->getAllocatedType();
    auto ResultV = TheHelper_.callFunction(
        TheModule,
        TaskI.getName(),
        ResultT,
        ArgVs);
    auto ReduceOp = dynamic_cast<const CudaReduceInfo*>(AbstractReduceOp);
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
    auto TaskStr = llvmString(TheContext_, TheModule, TaskI.getName());
    TheHelper_.callFunction(
      TheModule,
      "contra_cuda_launch_kernel",
      VoidType_,
      {TaskStr, RangeA, ExpandedArgsA});
  }


  //----------------------------------------------------------------------------
  // cleanup
  
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
      llvmValue<bool>(TheContext_, true)
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
      "contra_serial_field_create",
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
      "contra_serial_field_destroy",
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
      "contra_serial_accessor_write",
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
      "contra_serial_accessor_read",
      VoidType_,
      FunArgVs);

  return TheHelper_.load(ValueA);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void CudaTasker::destroyAccessor(
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
    Module &,
    const std::string &,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{
  return std::make_unique<CudaReduceInfo>(VarTs, ReduceTypes);
}


}
