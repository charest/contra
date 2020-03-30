#include "config.hpp"
#include "llvm_utils.hpp"

#include "codegen.hpp"
#include "errors.hpp"
#include "legion.hpp"

#include "legion/legion_c.h"

#include <vector>

namespace contra {

using namespace llvm;

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * createOpaqueType(const std::string & Name, LLVMContext & TheContext)
{
  auto OpaqueType = StructType::create( TheContext, Name );
  auto VoidPointerType = llvmType<void*>(TheContext);

  std::vector<Type*> members{ VoidPointerType }; 
  OpaqueType->setBody( members );

  return OpaqueType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * createTaskConfigOptionsType(const std::string & Name, LLVMContext & TheContext)
{
  auto BoolType = llvmType<bool>(TheContext);
  std::vector<Type*> members(4, BoolType);
  auto OptionsType = StructType::create( TheContext, members, Name );
  return OptionsType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * createTaskArgumentsType(const std::string & Name, LLVMContext & TheContext)
{
  auto VoidPointerType = llvmType<void*>(TheContext);
  auto SizeType = llvmType<std::size_t>(TheContext);
  std::vector<Type*> members = { VoidPointerType, SizeType };
  auto NewType = StructType::create( TheContext, members, Name );
  return NewType;
}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(Module &TheModule,
    const std::string & Name, Function* TaskF)
{
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto VoidT = llvmType<void>(TheContext_);
  auto SizeT = llvmType<std::size_t>(TheContext_);
  auto RealmIdT = llvmType<realm_id_t>(TheContext_);

  //----------------------------------------------------------------------------
  // Create task wrapper
  std::string TaskName = "__" + Name + "_task__";

  std::vector<Type *> WrapperArgTs =
    {VoidPtrT, SizeT, VoidPtrT, SizeT, RealmIdT};
  
  auto WrapperT = FunctionType::get(VoidT, WrapperArgTs, false);
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
  auto DataV = Builder_.CreateLoad(VoidPtrT, WrapperArgVs[0], "data");
  auto DataLenV = Builder_.CreateLoad(SizeT, WrapperArgVs[1], "datalen");
  //auto UserDataV = Builder_.CreateLoad(VoidPtrT, WrapperArgVs[2], "userdata");
  //auto UserLenV = Builder_.CreateLoad(SizeT, WrapperArgVs[3], "userlen");
  auto ProcIdV = Builder_.CreateLoad(RealmIdT, WrapperArgVs[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto TaskT = createOpaqueType("legion_task_t", TheContext_);
  auto TaskA = createEntryBlockAlloca(WrapperF, TaskT, "task.alloca");
 
  auto RegionT = createOpaqueType("legion_physical_region_t", TheContext_);
  auto RegionsT = RegionT->getPointerTo();
  auto RegionsA = createEntryBlockAlloca(WrapperF, RegionsT, "regions.alloca");
  auto NullV = Constant::getNullValue(RegionsT);
  Builder_.CreateStore(NullV, RegionsA);

  auto NumRegionsT = llvmType<std::uint32_t>(TheContext_); 
  auto NumRegionsA = createEntryBlockAlloca(WrapperF, NumRegionsT, "num_regions");
  auto ZeroV = llvmValue<std::uint32_t>(TheContext_, 0);
  Builder_.CreateStore(ZeroV, NumRegionsA);
 
  auto ContextT = createOpaqueType("legion_context_t", TheContext_);
  ContextAlloca_ = createEntryBlockAlloca(WrapperF, ContextT, "context.alloca");

  auto RuntimeT = createOpaqueType("legion_runtime_t", TheContext_);
  RuntimeAlloca_ = createEntryBlockAlloca(WrapperF, RuntimeT, "runtime.alloca");

  // args
  std::vector<Value*> PreambleArgVs = { DataV, DataLenV, ProcIdV,
    TaskA, RegionsA, NumRegionsA, ContextAlloca_, RuntimeAlloca_ };
  auto PreambleArgTs = llvmTypes(PreambleArgVs);
  
  auto PreambleT = FunctionType::get(VoidT, PreambleArgTs, false);
  auto PreambleF = TheModule.getOrInsertFunction("legion_task_preamble", PreambleT);
  
  Builder_.CreateCall(PreambleF, PreambleArgVs, "preamble");
  
  //----------------------------------------------------------------------------
  // Get task args

  auto TaskRV = load(TaskA, TheModule, "task");
  auto TaskGetArgsT = FunctionType::get(VoidPtrT, TaskRV->getType(), false);
  auto TaskGetArgsF = TheModule.getOrInsertFunction("legion_task_get_args", TaskGetArgsT);
  
  Value* TaskArgsV = Builder_.CreateCall(TaskGetArgsF, TaskRV, "args");
  
  auto TaskArgsA = createEntryBlockAlloca(WrapperF, VoidPtrT, "args.alloca");
  Builder_.CreateStore(TaskArgsV, TaskArgsA);
     
  //----------------------------------------------------------------------------
  // Argument sizes
  
  std::vector<Value*> ArgSizes;
  for (auto &Arg : TaskF->args())
    ArgSizes.emplace_back( getTypeSize<size_t>(Builder_, Arg.getType()) );
  
  //----------------------------------------------------------------------------
  // unpack user variables
    
  auto OffsetT = SizeT;
  auto OffsetA = createEntryBlockAlloca(WrapperF, OffsetT, "offset.alloca");
  ZeroV = Constant::getNullValue(OffsetT);
  Builder_.CreateStore(ZeroV, OffsetA);

  std::vector<AllocaInst*> TaskArgAs;

  ArgIdx = 0;
  for (auto &Arg : TaskF->args()) {
    // load
    auto OffsetV = Builder_.CreateLoad(OffsetT, OffsetA, "offset");
    // offset 
    auto ByteT = VoidPtrT->getPointerElementType();
    auto TaskArgsV = Builder_.CreateLoad(VoidPtrT, TaskArgsA, "args");
    auto ArgGEP = Builder_.CreateGEP(ByteT, TaskArgsV, OffsetV, "args.gep");
    // extract
    auto ArgT = Arg.getType();
    auto ArgN = Arg.getName().str() + ".alloca";
    auto ArgA = createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    TaskArgAs.emplace_back(ArgA);
    auto ArgPtrT = ArgT->getPointerTo();
    auto TheBlock = Builder_.GetInsertBlock();
    auto ArgC = CastInst::Create(CastInst::BitCast, ArgGEP, ArgPtrT, "casttmp", TheBlock);
    Builder_.CreateMemCpy(ArgA, 1, ArgC, 1, ArgSizes[ArgIdx]); 
    // increment
    auto NewOffsetV = Builder_.CreateAdd(OffsetV, ArgSizes[ArgIdx], "addoffset");
    Builder_.CreateStore( NewOffsetV, OffsetA );
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
  auto RuntimeT = RuntimeAlloca_->getType()->getPointerElementType();
  auto ContextT = ContextAlloca_->getType()->getPointerElementType();
  auto RuntimeV = Builder_.CreateLoad(RuntimeT, RuntimeAlloca_, "runtime");
  auto ContextV = Builder_.CreateLoad(ContextT, ContextAlloca_, "context");

  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto RetvalV = Constant::getNullValue(VoidPtrT);
  auto RetsizeV = llvmValue<std::size_t>(TheContext_, 0);

  // args
  std::vector<Value*> PostambleArgVs = { RuntimeV, ContextV, RetvalV, RetsizeV };
  sanitize(PostambleArgVs, TheModule);
  std::vector<Type*> PostambleArgTs = llvmTypes(PostambleArgVs);

  // call
  auto VoidT = llvmType<void>(TheContext_);
  auto PostambleT = FunctionType::get(VoidT, PostambleArgTs, false);
  auto PostambleF = TheModule.getOrInsertFunction("legion_task_postamble", PostambleT);
  
  Builder_.CreateCall(PostambleF, PostambleArgVs, "preamble");
  
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
  auto ExecSetT = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  auto ExecSetRT = reduceStruct(ExecSetT, TheModule);
  
  auto ExecT = FunctionType::get(ExecSetRT, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_execution_constraint_set_create", ExecT);
  
  Value* ExecSetRV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = createEntryBlockAlloca(TheFunction, ExecSetT, "execution_constraint.alloca");
  store(ExecSetRV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto VoidT = llvmType<void>(TheContext_);
  auto ProcIdV = llvmValue<legion_processor_kind_t>(TheContext_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidT, AddExecArgTs, false);
  auto AddExecF = TheModule.getOrInsertFunction(
      "legion_execution_constraint_set_add_processor_constraint", AddExecT);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetT = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  auto LayoutSetRT = reduceStruct(LayoutSetT, TheModule);

  auto LayoutT = FunctionType::get(LayoutSetRT, false);
  auto LayoutF = TheModule.getOrInsertFunction(
      "legion_task_layout_constraint_set_create", LayoutT);
  
  Value* LayoutSetRV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = createEntryBlockAlloca(TheFunction, LayoutSetT, "layout_constraint.alloca");
  store(LayoutSetRV, LayoutSetA);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigT = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  auto TaskConfigA = createEntryBlockAlloca(TheFunction, TaskConfigT, "options");
  auto BoolT = TaskConfigT->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto TaskIdVariantT = llvmType<legion_variant_id_t>(TheContext_);
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue<legion_task_id_t>(TheContext_, Task.getId());
  auto TaskIdVariantV = llvmValue<legion_variant_id_t>(TheContext_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrT);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  auto TrueV = Constant::getNullValue(BoolT);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, TrueV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreArgTs = llvmTypes(PreArgVs);
  auto PreRetT = TaskIdVariantT;

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
  auto ExecSetT = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  auto ExecSetRT = reduceStruct(ExecSetT, TheModule);
  
  auto ExecT = FunctionType::get(ExecSetRT, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_execution_constraint_set_create",
      ExecT);
  
  Value* ExecSetRV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = createEntryBlockAlloca(TheFunction, ExecSetT, "execution_constraint.alloca");
  store(ExecSetRV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto VoidT = llvmType<void>(TheContext_);
  auto ProcIdV = llvmValue<legion_processor_kind_t>(TheContext_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidT, AddExecArgTs, false);
  auto AddExecF = TheModule.getOrInsertFunction(
      "legion_execution_constraint_set_add_processor_constraint", AddExecT);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetT = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  auto LayoutSetRT = reduceStruct(LayoutSetT, TheModule);

  auto LayoutT = FunctionType::get(LayoutSetRT, false);
  auto LayoutF = TheModule.getOrInsertFunction(
      "legion_task_layout_constraint_set_create", LayoutT);
  
  Value* LayoutSetRV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = createEntryBlockAlloca(TheFunction, LayoutSetT, "layout_constraint.alloca");
  store(LayoutSetRV, LayoutSetA);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigT = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  auto TaskConfigA = createEntryBlockAlloca(TheFunction, TaskConfigT, "options");
  auto BoolT = TaskConfigT->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto TaskIdVariantT = llvmType<legion_variant_id_t>(TheContext_);
 
  auto TaskT = Task.getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue<legion_task_id_t>(TheContext_, Task.getId());
  auto TaskIdVariantV = llvmValue<legion_variant_id_t>(TheContext_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrT);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreArgTs = llvmTypes(PreArgVs);
  auto PreRetT = TaskIdVariantT;

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

  auto VoidT = llvmType<void>(TheContext_);
  
  auto TaskIdV = llvmValue<legion_task_id_t>(TheContext_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
  auto SetArgTs = llvmTypes(SetArgVs);

  auto SetT = FunctionType::get(VoidT, SetArgTs, false);
  auto SetF = TheModule.getOrInsertFunction(
      "legion_runtime_set_top_level_task_id", SetT);

  Builder_.CreateCall(SetF, SetArgVs, "set_top");
}
  
//==============================================================================
// start runtime
//==============================================================================
llvm::Value* LegionTasker::startRuntime(llvm::Module &TheModule, int Argc, char ** Argv)
{
  auto VoidPtrArrayT = llvmType<void*>(TheContext_)->getPointerTo();
  auto IntT = llvmType<int>(TheContext_);
  auto BoolT = llvmType<bool>(TheContext_);

  std::vector<Type*> StartArgTs = { IntT, VoidPtrArrayT, BoolT };
  auto StartT = FunctionType::get(IntT, StartArgTs, false);
  auto StartF = TheModule.getOrInsertFunction("legion_runtime_start", StartT);

  auto ArgcV = llvmValue<int_t>(TheContext_, Argc);
  auto ArgvV = Constant::getNullValue(VoidPtrArrayT);
  auto BackV = llvmValue<bool>(TheContext_, false);

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV, BackV };
  auto RetI = Builder_.CreateCall(StartF, StartArgVs, "start");
  return RetI;
}

//==============================================================================
// Launch a task
//==============================================================================
void LegionTasker::launch(Module &TheModule, const std::string & Name,
    const TaskInfo & TaskI, const std::vector<llvm::Value*> & ArgVs,
    const std::vector<llvm::Value*> & ArgSizes )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  auto TaskArgsT = createTaskArgumentsType("legion_task_argument_t", TheContext_);
  auto TaskArgsA = createEntryBlockAlloca(TheFunction, TaskArgsT, "args.alloca");

  //----------------------------------------------------------------------------
  std::vector<Value*> MemberIndices(2);
  MemberIndices[0] = ConstantInt::get(TheContext_, APInt(32, 0, true)); 

  auto createGEP = [&](int id, const std::string & Name) {
    MemberIndices[1] = ConstantInt::get(TheContext_, APInt(32, id, true));
    return Builder_.CreateGEP(TaskArgsT, TaskArgsA, MemberIndices, Name);
  };
  
  //----------------------------------------------------------------------------
  // First count sizes

  auto ArgSizeGEP = createGEP(1, "arglen");
  auto ArgSizeT = TaskArgsT->getElementType(1);
  Builder_.CreateStore( Constant::getNullValue(ArgSizeT), ArgSizeGEP );
  auto ArgSizeV = Builder_.CreateLoad(ArgSizeT, ArgSizeGEP);

  auto NumArgs = ArgVs.size();
  for (unsigned i=0; i<NumArgs; i++) {
    auto NewSizeV = Builder_.CreateAdd(ArgSizeV, ArgSizes[i], "addoffset");
    ArgSizeGEP = createGEP(1, "arglen");
    Builder_.CreateStore( NewSizeV, ArgSizeGEP );
    ArgSizeV = Builder_.CreateLoad(ArgSizeT, ArgSizeGEP);
  }
  
 
  //----------------------------------------------------------------------------
  // Allocate storate
  
  auto ByteT = llvmType<char>(TheContext_);
  auto TheBlock = Builder_.GetInsertBlock();
  auto OneV = llvmValue<std::size_t>(TheContext_, 1);
  auto SizeT = llvmType<std::size_t>(TheContext_);
  auto TmpA = Builder_.CreateAlloca(ByteT, nullptr); // not needed but InsertAtEnd
                                                     // doesnt work
  auto MallocI = CallInst::CreateMalloc(TmpA, SizeT, ByteT, ArgSizeV,
      OneV, nullptr, "args" );
  TmpA->eraseFromParent();
  
  auto ArgDataGEP = createGEP(0, "args");
  auto ArgDataT = TaskArgsT->getElementType(0);
  Builder_.CreateStore(MallocI, ArgDataGEP );
 
  //----------------------------------------------------------------------------
  // Copy args
  
  ArgSizeGEP = createGEP(1, "arglen");
  Builder_.CreateStore( Constant::getNullValue(ArgSizeT), ArgSizeGEP );

  for (unsigned i=0; i<NumArgs; i++) {
    auto ArgT = ArgVs[i]->getType();
    auto ArgPtrT = ArgT->getPointerTo();
    // copy argument into an alloca
    auto ArgA = createEntryBlockAlloca(TheFunction, ArgT, "tmpalloca");
    Builder_.CreateStore( ArgVs[i], ArgA );
    // load offset
    ArgSizeGEP = createGEP(1, "arglen");
    ArgSizeV = Builder_.CreateLoad(ArgSizeT, ArgSizeGEP);
    // copy
    ArgDataGEP = createGEP(0, "args");
    auto ArgDataPtrV = Builder_.CreateLoad(ArgDataT, ArgDataGEP, "args.ptr");
    auto OffsetArgDataPtrV = Builder_.CreateGEP(ArgDataPtrV, ArgSizeV, "args.offset");
    Builder_.CreateMemCpy(OffsetArgDataPtrV, 1, ArgA, 1, ArgSizes[i]); 
    // increment
    auto NewSizeV = Builder_.CreateAdd(ArgSizeV, ArgSizes[i], "addoffset");
    ArgSizeGEP = createGEP(1, "arglen");
    Builder_.CreateStore( NewSizeV, ArgSizeGEP );
    ArgSizeV = Builder_.CreateLoad(ArgSizeT, ArgSizeGEP);
  }

  
  //----------------------------------------------------------------------------
  // Predicate
    
  auto PredicateT = createOpaqueType("legion_predicate_t", TheContext_);
  auto PredicateRT = reduceStruct(PredicateT, TheModule);
  
  auto PredicateTrueT = FunctionType::get(PredicateRT, false);
  auto PredicateTrueF = TheModule.getOrInsertFunction(
      "legion_predicate_true", PredicateTrueT);
  
  Value* PredicateRV = Builder_.CreateCall(PredicateTrueF, None, "pred_true");
  
  auto PredicateA = createEntryBlockAlloca(TheFunction, PredicateT, "predicate.alloca");
  store(PredicateRV, PredicateA);
  
  //----------------------------------------------------------------------------
  // Launch
  
  auto MapperIdV = llvmValue<legion_mapper_id_t>(TheContext_, 0); 
  auto MappingTagIdV = llvmValue<legion_mapping_tag_id_t>(TheContext_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue<legion_task_id_t>(TheContext_, TaskI.getId());
  
  ArgDataGEP = createGEP(0, "args");
  auto ArgDataV = Builder_.CreateLoad(ArgDataT, ArgDataGEP, "args");
  
  ArgSizeGEP = createGEP(1, "arglen");
  ArgSizeV = Builder_.CreateLoad(ArgSizeT, ArgSizeGEP, "arglen");

  auto LauncherT = createOpaqueType("legion_task_launcher_t", TheContext_);
  auto LauncherRT = reduceStruct(LauncherT, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, ArgDataV, ArgSizeV, 
    PredicateV, MapperIdV, MappingTagIdV};
  auto LaunchArgTs = llvmTypes(LaunchArgVs);
  auto LaunchRetT = LauncherRT;

  auto LaunchT = FunctionType::get(LaunchRetT, LaunchArgTs, false);
  auto LaunchF = TheModule.getOrInsertFunction("legion_task_launcher_create", LaunchT);

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = createEntryBlockAlloca(TheFunction, LauncherT, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Execute
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextAlloca_, TheModule, "context");
  auto RuntimeV = load(RuntimeAlloca_, TheModule, "runtime");

  // args
  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };
  auto ExecArgTs = llvmTypes(ExecArgVs);
  
  auto FutureT = createOpaqueType("legion_future_t", TheContext_);
  auto FutureRT = reduceStruct(FutureT, TheModule);
  
  auto ExecT = FunctionType::get(FutureRT, ExecArgTs, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_task_launcher_execute", ExecT);
  
  auto FutureRV = Builder_.CreateCall(ExecF, ExecArgVs, "launcher_exec");
  auto FutureA = createEntryBlockAlloca(TheFunction, FutureT, "future.alloca");
  store(FutureRV, FutureA);

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");

  auto VoidT = llvmType<void>(TheContext_);

  auto DestroyT = FunctionType::get(VoidT, LauncherRT, false);
  auto DestroyF = TheModule.getOrInsertFunction("legion_task_launcher_destroy", DestroyT);

  Builder_.CreateCall(DestroyF, LauncherRV, "launcher_destroy");
  
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  ArgDataGEP = createGEP(0, "args");
  auto ArgDataPtrV = Builder_.CreateLoad(ArgDataT, ArgDataGEP, "args.ptr");
  TmpA = Builder_.CreateAlloca(VoidT, nullptr); // not needed but InsertAtEnd
                                                     // doesnt work
  CallInst::CreateFree(ArgDataPtrV, TmpA);
  TmpA->eraseFromParent();
}

}
