#include "serial.hpp"

#include "errors.hpp"

#include "librt/dopevector.hpp"
#include "utils/llvm_utils.hpp"

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
  auto NewType = StructType::create( TheContext_, members, "contra_serial_field_t" );
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
    VoidPtrType_,
    FieldType_->getPointerTo(),
    IndexPartitionType_->getPointerTo() };
  auto NewType = StructType::create( TheContext_, members, "contra_serial_accessor_t" );
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
void SerialTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
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
    const std::vector<Type*> & TaskArgTs)
{

  std::vector<std::string> WrapperArgNs = TaskArgNs;
    
  std::vector<Type*> WrapperArgTs;
  for (auto ArgT : TaskArgTs) {
    if (isRange(ArgT)) ArgT = IndexPartitionType_;
    WrapperArgTs.emplace_back(ArgT);
  }

  WrapperArgTs.emplace_back(IntType_);
  WrapperArgNs.emplace_back("index");

  auto WrapperT = FunctionType::get(VoidType_, WrapperArgTs, false);
  auto WrapperF = Function::Create(
      WrapperT,
      Function::ExternalLinkage,
      TaskName,
      &TheModule);
  
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

  auto GetRangeF = TheHelper_.createFunction(
      TheModule,
      "contra_serial_index_space_create_from_partition",
      VoidType_,
      GetRangeArgTs);
  
  auto IndexA = WrapperArgAs.back();
  auto IndexV = TheHelper_.load(IndexA);

  for (unsigned i=0; i<TaskArgNs.size(); i++) {
    if (isRange(TaskArgTs[i])) {
      auto ArgN = TaskArgNs[i];
      auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceType_, ArgN);
      Builder_.CreateCall(GetRangeF, {IndexV, WrapperArgAs[i], ArgA});
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
    bool HasReduction,
    int RedopId)
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

  for (unsigned i=0; i<NumArgs; i++) {
    if (!isField(ArgAs[i])) continue;
    auto FieldA = TheHelper_.getAsAlloca(ArgAs[i]);
    Value* IndexPartitionA = Constant::getNullValue(IndexPartitionType_->getPointerTo());
    if (PartAs[i]) IndexPartitionA = TheHelper_.getAsAlloca(PartAs[i]);
    auto AccessorA = TheHelper_.createEntryBlockAlloca(AccessorType_);
    std::vector<Value*> FunArgVs = {
      IndexSpaceA,
      IndexPartitionA,
      PartInfoA,
      FieldA,
      AccessorA};
    TheHelper_.callFunction(
        TheModule,
        "contra_serial_accessor_create",
        VoidType_,
        FunArgVs);
    ArgAs[i] = AccessorA;
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

  // set the current accessor partition
  for (auto ArgA : ArgAs) {
    if (isAccessor(ArgA)) {
      auto VarV = TheHelper_.load(VarA);
      TheHelper_.callFunction(
          TheModule,
          "contra_serial_accessor_set_current",
          VoidType_,
          {VarV, ArgA} );
    }
  }

  // CALL FUNCTION HERE
  ArgAs.emplace_back(VarA);

  std::vector<Value*> ArgVs;
  for (auto ArgA : ArgAs)
    ArgVs.emplace_back( TheHelper_.getAsValue(ArgA) );

  TheHelper_.callFunction(
      TheModule,
      TaskI.getName(),
      VoidType_,
      ArgVs);
      
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


  //----------------------------------------------------------------------------
  // cleanup
  
  destroyPartitions(TheModule, TempParts);
  destroyPartitionInfo(TheModule, PartInfoA);

  return nullptr;
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
AllocaInst* SerialTasker::createPartition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* IndexPartitionA,
    Value* ValueA)
{
  abort();
}

//==============================================================================
// Is this a field type
//==============================================================================
bool SerialTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldType_);
}


//==============================================================================
// Create a legion field
//==============================================================================
void SerialTasker::createField(
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

bool SerialTasker::isAccessor(Value* AccessorA) const
{
  auto AccessorT = AccessorA->getType();
  if (isa<AllocaInst>(AccessorA)) AccessorT = AccessorT->getPointerElementType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void SerialTasker::storeAccessor(
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
Value* SerialTasker::loadAccessor(
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

bool SerialTasker::isPartition(Value* PartA) const
{
  auto PartT = PartA->getType();
  if (isa<AllocaInst>(PartA)) PartT = PartT->getPointerElementType();
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

}
