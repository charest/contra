#include "serial.hpp"

#include "errors.hpp"

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
  FieldDataType_ = createFieldDataType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * SerialTasker::createFieldDataType()
{
  std::vector<Type*> members = { IntType_, VoidPtrType_ };
  auto NewType = StructType::create( TheContext_, members, "contra_serial_field_t" );
  return NewType;
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

  std::vector<Type*> WrapperArgTs = TaskArgTs;
  std::vector<std::string> WrapperArgNs = TaskArgNs;

  WrapperArgTs.emplace_back(IntType_);
  WrapperArgNs.emplace_back("index");

  auto WrapperT = FunctionType::get(VoidType_, WrapperArgTs, false);
  auto WrapperF = Function::Create(WrapperT, Function::ExternalLinkage,
      TaskName, &TheModule);
  
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

  auto IndexA = WrapperArgAs.back();
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
  //if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);

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
      //std::vector<Value*> FunArgVs = {
      //  IndexSpaceA,
      //  IndexPartitionA,
      //  PartInfoA};
      //TheHelper_.callFunction(
      //    TheModule,
      //    "contra_serial_register_index_partition",
      //    VoidType_,
      //    FunArgVs);
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
  
  //destroyPartitions(TheModule, TempParts);
}

//==============================================================================
// Is this a field type
//==============================================================================
bool SerialTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldDataType_);
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

}
