#include "config.hpp"

#include "codegen.hpp"
#include "errors.hpp"
#include "legion.hpp"
#include "utils/llvm_utils.hpp"

#include "legion/legion_c.h"

#include <vector>

namespace contra {

using namespace llvm;
using namespace utils;

////////////////////////////////////////////////////////////////////////////////
// Legion tasker
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Constructor
//==============================================================================
LegionTasker::LegionTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext) :
  AbstractTasker(TheBuilder, TheContext)
{
  VoidPtrType_ = llvmType<void*>(TheContext_);
  ByteType_ = VoidPtrType_->getPointerElementType();
  VoidType_ = llvmType<void>(TheContext_);
  SizeType_ = llvmType<std::size_t>(TheContext_);
  Int32Type_ = llvmType<int>(TheContext_);
  BoolType_ = llvmType<bool>(TheContext_);
  RealmIdType_ = llvmType<realm_id_t>(TheContext_);
  NumRegionsType_ = llvmType<std::uint32_t>(TheContext_); 
  TaskVariantIdType_ = llvmType<legion_variant_id_t>(TheContext_);
  TaskIdType_ = llvmType<legion_task_id_t>(TheContext_);
  ProcIdType_ = llvmType<legion_processor_kind_t>(TheContext_);
  MapperIdType_ = llvmType<legion_mapper_id_t>(TheContext_); 
  MappingTagIdType_ = llvmType<legion_mapping_tag_id_t>(TheContext_);

  TaskType_ = createOpaqueType("legion_task_t", TheContext_);
  RegionType_ = createOpaqueType("legion_physical_region_t", TheContext_);
  ContextType_ = createOpaqueType("legion_context_t", TheContext_);
  RuntimeType_ = createOpaqueType("legion_runtime_t", TheContext_);
  ExecSetType_ = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  LayoutSetType_ = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  PredicateType_ = createOpaqueType("legion_predicate_t", TheContext_);
  LauncherType_ = createOpaqueType("legion_task_launcher_t", TheContext_);
  FutureType_ = createOpaqueType("legion_future_t", TheContext_);
  TaskConfigType_ = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  TaskArgsType_ = createTaskArgumentsType("legion_task_argument_t", TheContext_);
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createOpaqueType(
    const std::string & Name, LLVMContext & TheContext)
{
  auto OpaqueType = StructType::create( TheContext, Name );

  std::vector<Type*> members{ VoidPtrType_ }; 
  OpaqueType->setBody( members );

  return OpaqueType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createTaskConfigOptionsType(
    const std::string & Name, LLVMContext & TheContext)
{
  std::vector<Type*> members(4, BoolType_);
  auto OptionsType = StructType::create( TheContext, members, Name );
  return OptionsType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createTaskArgumentsType(
    const std::string & Name, LLVMContext & TheContext)
{
  std::vector<Type*> members = { VoidPtrType_, SizeType_ };
  auto NewType = StructType::create( TheContext, members, Name );
  return NewType;
}


//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(Module &TheModule,
    const std::string & Name, Function* TaskF)
{
  //----------------------------------------------------------------------------
  // Create task wrapper
  std::string TaskName = "__" + Name + "_task__";

  std::vector<Type *> WrapperArgTs =
    {VoidPtrType_, SizeType_, VoidPtrType_, SizeType_, RealmIdType_};
  
  auto WrapperT = FunctionType::get(VoidType_, WrapperArgTs, false);
  auto WrapperF = Function::Create(WrapperT, Function::ExternalLinkage,
      TaskName, &TheModule);
 
  auto Arg = WrapperF->arg_begin();
  Arg->setName("data");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("datalen");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("userdata");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("userlen");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("procid");
  Arg->addAttr(Attribute::ReadOnly);

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", WrapperF);
  Builder_.SetInsertPoint(BB);

  // allocate arguments
  std::vector<Value*> WrapperArgVs;
  WrapperArgVs.reserve(WrapperArgTs.size());

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgVs.emplace_back(Alloca);
    ArgIdx++;
  }

  // loads
  auto DataV = Builder_.CreateLoad(VoidPtrType_, WrapperArgVs[0], "data");
  auto DataLenV = Builder_.CreateLoad(SizeType_, WrapperArgVs[1], "datalen");
  //auto UserDataV = Builder_.CreateLoad(VoidPtrType_, WrapperArgVs[2], "userdata");
  //auto UserLenV = Builder_.CreateLoad(SizeType_, WrapperArgVs[3], "userlen");
  auto ProcIdV = Builder_.CreateLoad(RealmIdType_, WrapperArgVs[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto TaskA = createEntryBlockAlloca(WrapperF, TaskType_, "task.alloca");
 
  auto RegionsT = RegionType_->getPointerTo();
  auto RegionsA = createEntryBlockAlloca(WrapperF, RegionsT, "regions.alloca");
  auto NullV = Constant::getNullValue(RegionsT);
  Builder_.CreateStore(NullV, RegionsA);

  auto NumRegionsA = createEntryBlockAlloca(WrapperF, NumRegionsType_, "num_regions");
  auto ZeroV = llvmValue(TheContext_, NumRegionsType_, 0);
  Builder_.CreateStore(ZeroV, NumRegionsA);
 
  ContextAlloca_ = createEntryBlockAlloca(WrapperF, ContextType_, "context.alloca");

  RuntimeAlloca_ = createEntryBlockAlloca(WrapperF, RuntimeType_, "runtime.alloca");

  // args
  std::vector<Value*> PreambleArgVs = { DataV, DataLenV, ProcIdV,
    TaskA, RegionsA, NumRegionsA, ContextAlloca_, RuntimeAlloca_ };
  auto PreambleArgTs = llvmTypes(PreambleArgVs);
  
  auto PreambleT = FunctionType::get(VoidType_, PreambleArgTs, false);
  auto PreambleF = TheModule.getOrInsertFunction("legion_task_preamble", PreambleT);
  
  Builder_.CreateCall(PreambleF, PreambleArgVs, "preamble");
  
  //----------------------------------------------------------------------------
  // Get task args

  auto TaskRV = load(TaskA, TheModule, "task");
  auto TaskGetArgsT = FunctionType::get(VoidPtrType_, TaskRV->getType(), false);
  auto TaskGetArgsF = TheModule.getOrInsertFunction("legion_task_get_args", TaskGetArgsT);
  
  Value* TaskArgsV = Builder_.CreateCall(TaskGetArgsF, TaskRV, "args");
  
  auto TaskArgsA = createEntryBlockAlloca(WrapperF, VoidPtrType_, "args.alloca");
  Builder_.CreateStore(TaskArgsV, TaskArgsA);
     
  //----------------------------------------------------------------------------
  // Argument sizes
  
  std::vector<Value*> ArgSizes;
  for (auto &Arg : TaskF->args())
    ArgSizes.emplace_back( getTypeSize<size_t>(Builder_, Arg.getType()) );
  
  //----------------------------------------------------------------------------
  // unpack user variables
    
  auto OffsetT = SizeType_;
  auto OffsetA = createEntryBlockAlloca(WrapperF, OffsetT, "offset.alloca");
  ZeroV = Constant::getNullValue(OffsetT);
  Builder_.CreateStore(ZeroV, OffsetA);

  std::vector<AllocaInst*> TaskArgAs;

  ArgIdx = 0;
  for (auto &Arg : TaskF->args()) {
    // offset 
    auto ArgGEP = offsetPointer(TaskArgsA, OffsetA, "args");
    // extract
    auto ArgT = Arg.getType();
    auto ArgN = Arg.getName().str() + ".alloca";
    auto ArgA = createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    TaskArgAs.emplace_back(ArgA);
    // copy
    memCopy(ArgGEP, ArgA, ArgSizes[ArgIdx]);
    // increment
    increment(OffsetA, ArgSizes[ArgIdx], "offset");
    ArgIdx++;
  }

  //----------------------------------------------------------------------------
  // Function body
  return {WrapperF, TaskArgAs}; 
}
  
//==============================================================================
// Create the function wrapper
//==============================================================================
void LegionTasker::taskPostamble(Module &TheModule, Value* ResultV)
{

  // temporaries
  auto RuntimeV = Builder_.CreateLoad(RuntimeType_, RuntimeAlloca_, "runtime");
  auto ContextV = Builder_.CreateLoad(ContextType_, ContextAlloca_, "context");
  
  Value* RetvalV = Constant::getNullValue(VoidPtrType_);
  Value* RetsizeV = llvmValue<std::size_t>(TheContext_, 0);

  AllocaInst* RetvalA;
  auto RetvalT = VoidPtrType_;

  
  //----------------------------------------------------------------------------
  // Have return value
  if (ResultV) {
    
    auto TheFunction = Builder_.GetInsertBlock()->getParent();
    
    // store result
    auto ResultT = ResultV->getType();
    auto ResultA = createEntryBlockAlloca(TheFunction, ResultT, "result");
    Builder_.CreateStore( ResultV, ResultA );

    // return size
    auto RetsizeT = RetsizeV->getType();
    RetsizeV = getTypeSize(Builder_, ResultT, RetsizeT);
    auto RetsizeA = createEntryBlockAlloca(TheFunction, RetsizeT, "retsize");
    Builder_.CreateStore( RetsizeV, RetsizeA );

    // allocate space for return value
    RetsizeV = Builder_.CreateLoad(RetsizeT, RetsizeA);
    
    auto TmpA = Builder_.CreateAlloca(ByteType_, nullptr); // not needed but InsertAtEnd doesnt work
    auto MallocI = CallInst::CreateMalloc(TmpA, RetsizeT, ByteType_, RetsizeV,
        nullptr, nullptr, "retval" );
    TmpA->eraseFromParent();

    RetvalA = createEntryBlockAlloca(TheFunction, RetvalT, "retval");
    Builder_.CreateStore(MallocI, RetvalA );

    // copy data
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    RetsizeV = Builder_.CreateLoad(RetsizeT, RetsizeA);
    Builder_.CreateMemCpy(RetvalV, 1, ResultA, 1, RetsizeV); 


    // final loads
    RetsizeV = Builder_.CreateLoad(RetsizeT, RetsizeA);
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    
  }

  //----------------------------------------------------------------------------
  // Call postable

  // args
  std::vector<Value*> PostambleArgVs = { RuntimeV, ContextV, RetvalV, RetsizeV };
  sanitize(PostambleArgVs, TheModule);
  std::vector<Type*> PostambleArgTs = llvmTypes(PostambleArgVs);

  // call
  auto PostambleT = FunctionType::get(VoidType_, PostambleArgTs, false);
  auto PostambleF = TheModule.getOrInsertFunction("legion_task_postamble", PostambleT);
  
  Builder_.CreateCall(PostambleF, PostambleArgVs, "preamble");
  
  //----------------------------------------------------------------------------
  // Free memory
  if (ResultV) {
    auto RetvalT = RetvalV->getType();
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    auto TmpA = Builder_.CreateAlloca(VoidType_, nullptr); // not needed but InsertAtEnd doesnt work
    CallInst::CreateFree(RetvalV, TmpA);
    TmpA->eraseFromParent();
  }

  
  reset();
}

//==============================================================================
// Postregister tasks
//==============================================================================
void LegionTasker::postregisterTask(llvm::Module &TheModule, const std::string & Name,
    const TaskInfo & Task )
{
  // get current insertion point
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  //----------------------------------------------------------------------------
  // execution_constraint_set
  auto ExecSetRT = reduceStruct(ExecSetType_, TheModule);
  
  auto ExecT = FunctionType::get(ExecSetRT, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_execution_constraint_set_create", ExecT);
  
  Value* ExecSetRV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = createEntryBlockAlloca(TheFunction, ExecSetType_, "execution_constraint.alloca");
  store(ExecSetRV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto ProcIdV = llvmValue(TheContext_, ProcIdType_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidType_, AddExecArgTs, false);
  auto AddExecF = TheModule.getOrInsertFunction(
      "legion_execution_constraint_set_add_processor_constraint", AddExecT);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetRT = reduceStruct(LayoutSetType_, TheModule);

  auto LayoutT = FunctionType::get(LayoutSetRT, false);
  auto LayoutF = TheModule.getOrInsertFunction(
      "legion_task_layout_constraint_set_create", LayoutT);
  
  Value* LayoutSetRV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = createEntryBlockAlloca(TheFunction, LayoutSetType_, "layout_constraint.alloca");
  store(LayoutSetRV, LayoutSetA);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigA = createEntryBlockAlloca(TheFunction, TaskConfigType_, "options");
  auto BoolT = TaskConfigType_->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(TheContext_, TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(TheContext_, TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  auto TrueV = Constant::getNullValue(BoolT);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, TrueV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreArgTs = llvmTypes(PreArgVs);
  auto PreRetT = TaskVariantIdType_;

  auto PreT = FunctionType::get(PreRetT, PreArgTs, false);
  auto PreF = TheModule.getOrInsertFunction(
      "legion_runtime_register_task_variant_fnptr", PreT);

  TaskIdV = Builder_.CreateCall(PreF, PreArgVs, "task_variant_id");
}

//==============================================================================
// Preregister tasks
//==============================================================================
void LegionTasker::preregisterTask(llvm::Module &TheModule, const std::string & Name,
    const TaskInfo & Task )
{
  // get current insertion point
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  //----------------------------------------------------------------------------
  // execution_constraint_set
  auto ExecSetRT = reduceStruct(ExecSetType_, TheModule);
  
  auto ExecT = FunctionType::get(ExecSetRT, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_execution_constraint_set_create",
      ExecT);
  
  Value* ExecSetRV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = createEntryBlockAlloca(TheFunction, ExecSetType_, "execution_constraint.alloca");
  store(ExecSetRV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto ProcIdV = llvmValue(TheContext_, ProcIdType_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidType_, AddExecArgTs, false);
  auto AddExecF = TheModule.getOrInsertFunction(
      "legion_execution_constraint_set_add_processor_constraint", AddExecT);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetRT = reduceStruct(LayoutSetType_, TheModule);

  auto LayoutT = FunctionType::get(LayoutSetRT, false);
  auto LayoutF = TheModule.getOrInsertFunction(
      "legion_task_layout_constraint_set_create", LayoutT);
  
  Value* LayoutSetRV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = createEntryBlockAlloca(TheFunction, LayoutSetType_, "layout_constraint.alloca");
  store(LayoutSetRV, LayoutSetA);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigA = createEntryBlockAlloca(TheFunction, TaskConfigType_, "options");
  auto BoolT = TaskConfigType_->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(TheContext_, TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(TheContext_, TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreArgTs = llvmTypes(PreArgVs);
  auto PreRetT = TaskVariantIdType_;

  auto PreT = FunctionType::get(PreRetT, PreArgTs, false);
  auto PreF = TheModule.getOrInsertFunction(
      "legion_runtime_preregister_task_variant_fnptr", PreT);

  TaskIdV = Builder_.CreateCall(PreF, PreArgVs, "task_variant_id");
}
  
//==============================================================================
// Set top level task
//==============================================================================
void LegionTasker::setTopLevelTask(llvm::Module &TheModule, int TaskId )
{

  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
  auto SetArgTs = llvmTypes(SetArgVs);

  auto SetT = FunctionType::get(VoidType_, SetArgTs, false);
  auto SetF = TheModule.getOrInsertFunction(
      "legion_runtime_set_top_level_task_id", SetT);

  Builder_.CreateCall(SetF, SetArgVs, "set_top");
}
  
//==============================================================================
// start runtime
//==============================================================================
llvm::Value* LegionTasker::startRuntime(llvm::Module &TheModule, int Argc, char ** Argv)
{
  auto VoidPtrArrayT = VoidPtrType_->getPointerTo();

  std::vector<Type*> StartArgTs = { Int32Type_, VoidPtrArrayT, BoolType_ };
  auto StartT = FunctionType::get(Int32Type_, StartArgTs, false);
  auto StartF = TheModule.getOrInsertFunction("legion_runtime_start", StartT);

  auto ArgcV = llvmValue(TheContext_, Int32Type_, Argc);
  auto ArgvV = Constant::getNullValue(VoidPtrArrayT);
  auto BackV = llvmValue(TheContext_, BoolType_, false);

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV, BackV };
  auto RetI = Builder_.CreateCall(StartF, StartArgVs, "start");
  return RetI;
}

//==============================================================================
// Launch a task
//==============================================================================
llvm::Value* LegionTasker::launch(Module &TheModule, const std::string & Name,
    int TaskId, const std::vector<llvm::Value*> & ArgVs,
    const std::vector<llvm::Value*> & ArgSizes )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  auto TaskArgsA = createEntryBlockAlloca(TheFunction, TaskArgsType_, "args.alloca");
  
  //----------------------------------------------------------------------------
  // First count sizes

  auto ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
  auto ArgSizeT = TaskArgsType_->getElementType(1);
  Builder_.CreateStore( Constant::getNullValue(ArgSizeT), ArgSizeGEP );

  auto NumArgs = ArgVs.size();
  for (unsigned i=0; i<NumArgs; i++) {
    ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
    increment(ArgSizeGEP, ArgSizes[i], "addoffset");
  }
  
 
  //----------------------------------------------------------------------------
  // Allocate storate
 
  auto ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");
  auto TmpA = Builder_.CreateAlloca(ByteType_, nullptr); // not needed but InsertAtEnd doesnt work
  auto MallocI = CallInst::CreateMalloc(TmpA, SizeType_, ByteType_, ArgSizeV,
      nullptr, nullptr, "args" );
  TmpA->eraseFromParent();
  storeStructMember(MallocI, TaskArgsA, 0, "args");
 
  //----------------------------------------------------------------------------
  // Copy args
  
  storeStructMember( Constant::getNullValue(ArgSizeT), TaskArgsA, 1, "arglen");
  
  for (unsigned i=0; i<NumArgs; i++) {
    auto ArgT = ArgVs[i]->getType();
    // copy argument into an alloca
    auto ArgA = createEntryBlockAlloca(TheFunction, ArgT, "tmpalloca");
    Builder_.CreateStore( ArgVs[i], ArgA );
    // load offset
    ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");
    // copy
    auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
    auto OffsetArgDataPtrV = Builder_.CreateGEP(ArgDataPtrV, ArgSizeV, "args.offset");
    Builder_.CreateMemCpy(OffsetArgDataPtrV, 1, ArgA, 1, ArgSizes[i]); 
    // increment
    ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
    increment(ArgSizeGEP, ArgSizes[i], "addoffset");
  }

  
  //----------------------------------------------------------------------------
  // Predicate
    
  auto PredicateRT = reduceStruct(PredicateType_, TheModule);
  
  auto PredicateTrueT = FunctionType::get(PredicateRT, false);
  auto PredicateTrueF = TheModule.getOrInsertFunction(
      "legion_predicate_true", PredicateTrueT);
  
  Value* PredicateRV = Builder_.CreateCall(PredicateTrueF, None, "pred_true");
  
  auto PredicateA = createEntryBlockAlloca(TheFunction, PredicateType_, "predicate.alloca");
  store(PredicateRV, PredicateA);
  
  //----------------------------------------------------------------------------
  // Launch
 
  auto MapperIdV = llvmValue(TheContext_, MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(TheContext_, MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");

  auto LauncherRT = reduceStruct(LauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, ArgDataPtrV, ArgSizeV, 
    PredicateV, MapperIdV, MappingTagIdV};
  auto LaunchArgTs = llvmTypes(LaunchArgVs);
  auto LaunchRetT = LauncherRT;

  auto LaunchT = FunctionType::get(LaunchRetT, LaunchArgTs, false);
  auto LaunchF = TheModule.getOrInsertFunction("legion_task_launcher_create", LaunchT);

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = createEntryBlockAlloca(TheFunction, LauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Execute
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextAlloca_, TheModule, "context");
  auto RuntimeV = load(RuntimeAlloca_, TheModule, "runtime");

  // args
  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };
  auto ExecArgTs = llvmTypes(ExecArgVs);
  
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  
  auto ExecT = FunctionType::get(FutureRT, ExecArgTs, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_task_launcher_execute", ExecT);
  
  auto FutureRV = Builder_.CreateCall(ExecF, ExecArgVs, "launcher_exec");
  auto FutureA = createEntryBlockAlloca(TheFunction, FutureType_, "future.alloca");
  store(FutureRV, FutureA);

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");

  auto DestroyT = FunctionType::get(VoidType_, LauncherRT, false);
  auto DestroyF = TheModule.getOrInsertFunction("legion_task_launcher_destroy", DestroyT);

  Builder_.CreateCall(DestroyF, LauncherRV, "launcher_destroy");
  
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  TmpA = Builder_.CreateAlloca(VoidType_, nullptr); // not needed but InsertAtEnd doesnt work
  CallInst::CreateFree(ArgDataPtrV, TmpA);
  TmpA->eraseFromParent();

  return FutureA;
}

//==============================================================================
// get a future value
//==============================================================================
llvm::Value* LegionTasker::getFuture(Module &TheModule, llvm::Value* FutureA,
    llvm::Type *DataT, llvm::Value* DataSizeV)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // args
  auto FutureRV = load(FutureA, TheModule, "future");
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto GetFutureT = FunctionType::get(VoidPtrType_, FutureRT, false);
  auto GetFutureF = TheModule.getOrInsertFunction("legion_future_get_untyped_pointer",
      GetFutureT);

  Value* DataPtrV = Builder_.CreateCall(GetFutureF, FutureRV, "future");
  auto DataPtrA = createEntryBlockAlloca(TheFunction, VoidPtrType_, "future.alloca");
  Builder_.CreateStore(DataPtrV, DataPtrA);

  DataPtrV = Builder_.CreateLoad(VoidPtrType_, DataPtrA);
  auto DataA = createEntryBlockAlloca(TheFunction, DataT);
  memCopy(DataPtrV, DataA, DataSizeV);

  return DataA;
}

}
