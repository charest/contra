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
Function* LegionTasker::wrapTask(Module &TheModule, const std::string & Name,
    Function* TaskF)
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

  auto TmpB = CodeGen::createBuilder(WrapperF);

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto Alloca = TmpB.CreateAlloca(ArgT, nullptr, Arg.getName()+".alloca");
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
  auto TaskA = TmpB.CreateAlloca(TaskT, nullptr, "task.alloca");
 
  auto RegionT = createOpaqueType("legion_physical_region_t", TheContext_);
  auto RegionsT = RegionT->getPointerTo();
  auto RegionsA = TmpB.CreateAlloca(RegionsT, nullptr, "regions.alloca");
  auto NullV = Constant::getNullValue(RegionsT);
  Builder_.CreateStore(NullV, RegionsA);

  auto NumRegionsT = llvmType<std::uint32_t>(TheContext_); 
  auto NumRegionsA = TmpB.CreateAlloca(NumRegionsT, nullptr, "num_regions");
  auto ZeroV = llvmValue<std::uint32_t>(TheContext_, 0);
  Builder_.CreateStore(ZeroV, NumRegionsA);
 
  auto ContextT = createOpaqueType("legion_context_t", TheContext_);
  ContextAlloca_ = TmpB.CreateAlloca(ContextT, nullptr, "context.alloca");

  auto RuntimeT = createOpaqueType("legion_runtime_t", TheContext_);
  RuntimeAlloca_ = TmpB.CreateAlloca(RuntimeT, nullptr, "runtime.alloca");

  // args
  std::vector<Value*> PreambleArgVs = { DataV, DataLenV, ProcIdV,
    TaskA, RegionsA, NumRegionsA, ContextAlloca_, RuntimeAlloca_ };
  auto PreambleArgTs = llvmTypes(PreambleArgVs);
  
  auto PreambleT = FunctionType::get(VoidT, PreambleArgTs, false);
  auto PreambleF = Function::Create(PreambleT, Function::InternalLinkage,
      "legion_task_preamble", &TheModule);
  
  Builder_.CreateCall(PreambleF, PreambleArgVs, "preamble");

  //----------------------------------------------------------------------------
  // extrat user variables
  
  //auto UserDataV;
  //auto UserLenV;

  auto DL = std::make_unique<DataLayout>(&TheModule);
  ArgIdx = 0;

  for (auto &Arg : TaskF->args()) {
    auto ArgT = Arg.getType();
    
    //legion_task_get_args(task)
    ArgIdx++;
  }


  std::vector<Value*> TaskArgVs;
  
  //----------------------------------------------------------------------------
  // call users actual function
  Builder_.CreateCall(TaskF, TaskArgVs, "calltmp");

  //----------------------------------------------------------------------------
  // Postable
 
  // temporaries
  auto RuntimeV = Builder_.CreateLoad(RuntimeT, RuntimeAlloca_, "runtime");
  auto ContextV = Builder_.CreateLoad(ContextT, ContextAlloca_, "context");

  auto RetvalV = Constant::getNullValue(VoidPtrT);
  auto RetsizeV = llvmValue<std::size_t>(TheContext_, 0);

  // args
  std::vector<Value*> PostambleArgVs = { RuntimeV, ContextV, RetvalV, RetsizeV };
  sanitize(PostambleArgVs, TheModule);
  std::vector<Type*> PostambleArgTs = llvmTypes(PostambleArgVs);

  // call
  auto PostambleT = FunctionType::get(VoidT, PostambleArgTs, false);
  auto PostambleF = Function::Create(PostambleT, Function::InternalLinkage,
      "legion_task_postamble", &TheModule);
  
  Builder_.CreateCall(PostambleF, PostambleArgVs, "preamble");
  
  //----------------------------------------------------------------------------
  // function retuns void
  Builder_.CreateRetVoid();

  reset();

  return WrapperF;
}

//==============================================================================
// Postregister tasks
//==============================================================================
void LegionTasker::postregisterTask(llvm::Module &TheModule, const std::string & Name,
    const TaskInfo & Task )
{
  // get current insertion point
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto TmpB = CodeGen::createBuilder(TheFunction);

  //----------------------------------------------------------------------------
  // execution_constraint_set
  auto ExecSetT = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  auto ExecSetRT = reduceStruct(ExecSetT, TheModule);
  
  auto ExecT = FunctionType::get(ExecSetRT, false);
  auto ExecF = Function::Create(ExecT, Function::InternalLinkage,
      "legion_execution_constraint_set_create", &TheModule);
  
  Value* ExecSetRV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = TmpB.CreateAlloca(ExecSetT, nullptr, "execution_constraint.alloca");
  store(ExecSetRV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto VoidT = llvmType<void>(TheContext_);
  auto ProcIdV = llvmValue<legion_processor_kind_t>(TheContext_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidT, AddExecArgTs, false);
  auto AddExecF = Function::Create(AddExecT, Function::InternalLinkage,
      "legion_execution_constraint_set_add_processor_constraint", &TheModule);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetT = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  auto LayoutSetRT = reduceStruct(LayoutSetT, TheModule);

  auto LayoutT = FunctionType::get(LayoutSetRT, false);
  auto LayoutF = Function::Create(LayoutT, Function::InternalLinkage,
      "legion_task_layout_constraint_set_create", &TheModule);
  
  Value* LayoutSetRV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = TmpB.CreateAlloca(LayoutSetT, nullptr, "layout_constraint.alloca");
  store(LayoutSetRV, LayoutSetA);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigT = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  auto TaskConfigA = TmpB.CreateAlloca(TaskConfigT, nullptr, "options");
  auto BoolT = TaskConfigT->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto TaskIdVariantT = llvmType<legion_variant_id_t>(TheContext_);
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = Function::Create(TaskT, Function::InternalLinkage, Task.getName(), &TheModule);
  
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
  auto PreF = Function::Create(PreT, Function::InternalLinkage,
      "legion_runtime_register_task_variant_fnptr", &TheModule);

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
  auto TmpB = CodeGen::createBuilder(TheFunction);

  //----------------------------------------------------------------------------
  // execution_constraint_set
  auto ExecSetT = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  auto ExecSetRT = reduceStruct(ExecSetT, TheModule);
  
  auto ExecT = FunctionType::get(ExecSetRT, false);
  auto ExecF = Function::Create(ExecT, Function::InternalLinkage,
      "legion_execution_constraint_set_create", &TheModule);
  
  Value* ExecSetRV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = TmpB.CreateAlloca(ExecSetT, nullptr, "execution_constraint.alloca");
  store(ExecSetRV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto VoidT = llvmType<void>(TheContext_);
  auto ProcIdV = llvmValue<legion_processor_kind_t>(TheContext_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidT, AddExecArgTs, false);
  auto AddExecF = Function::Create(AddExecT, Function::InternalLinkage,
      "legion_execution_constraint_set_add_processor_constraint", &TheModule);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetT = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  auto LayoutSetRT = reduceStruct(LayoutSetT, TheModule);

  auto LayoutT = FunctionType::get(LayoutSetRT, false);
  auto LayoutF = Function::Create(LayoutT, Function::InternalLinkage,
      "legion_task_layout_constraint_set_create", &TheModule);
  
  Value* LayoutSetRV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = TmpB.CreateAlloca(LayoutSetT, nullptr, "layout_constraint.alloca");
  store(LayoutSetRV, LayoutSetA);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigT = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  auto TaskConfigA = TmpB.CreateAlloca(TaskConfigT, nullptr, "options");
  auto BoolT = TaskConfigT->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto TaskIdVariantT = llvmType<legion_variant_id_t>(TheContext_);
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = Function::Create(TaskT, Function::InternalLinkage, Task.getName(), &TheModule);
  
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
  auto PreF = Function::Create(PreT, Function::InternalLinkage,
      "legion_runtime_preregister_task_variant_fnptr", &TheModule);

  TaskIdV = Builder_.CreateCall(PreF, PreArgVs, "task_variant_id");
}
  
//==============================================================================
// Set top level task
//==============================================================================
void LegionTasker::setTop(llvm::Module &TheModule, int TaskId )
{

  auto VoidT = llvmType<void>(TheContext_);
  
  auto TaskIdV = llvmValue<legion_task_id_t>(TheContext_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
  auto SetArgTs = llvmTypes(SetArgVs);

  auto SetT = FunctionType::get(VoidT, SetArgTs, false);
  auto SetF = Function::Create(SetT, Function::InternalLinkage,
      "legion_runtime_set_top_level_task_id", &TheModule);

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
  auto StartF = Function::Create(StartT, Function::InternalLinkage,
      "legion_runtime_start", &TheModule);

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
  auto TmpB = CodeGen::createBuilder(TheFunction);
  
  auto TaskArgsT = createTaskArgumentsType("legion_task_argument_t", TheContext_);
  auto TaskArgsA = TmpB.CreateAlloca(TaskArgsT, nullptr, "args.alloca");

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
  auto MallocI = CallInst::CreateMalloc(TheBlock, ByteT, ByteT, ArgSizeV,
      nullptr, nullptr, "args" );
  
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
    // load offset
    ArgSizeGEP = createGEP(1, "arglen");
    ArgSizeV = Builder_.CreateLoad(ArgSizeT, ArgSizeGEP);
    // copy
    ArgDataGEP = createGEP(0, "args");
    auto ArgDataPtrV = Builder_.CreateLoad(ArgDataT, ArgDataGEP, "args.ptr");
    auto OffsetArgDataPtrV = Builder_.CreateGEP(ArgDataPtrV, ArgSizeV, "args.offset");
    auto ArgInt = llvmValue<intptr_t>(TheContext_, (intptr_t)(ArgVs[i]));
    auto ArgPtrV = CastInst::Create(Instruction::IntToPtr, ArgInt, ArgPtrT, "argptr", TheBlock);
    Builder_.CreateMemCpy(OffsetArgDataPtrV, 1, ArgPtrV, 1, ArgSizes[i]); 
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
  auto PredicateTrueF = Function::Create(PredicateTrueT, Function::InternalLinkage,
      "legion_predicate_true", &TheModule);
  
  Value* PredicateRV = Builder_.CreateCall(PredicateTrueF, None, "pred_true");
  
  auto PredicateA = TmpB.CreateAlloca(PredicateT, nullptr, "predicate.alloca");
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
  auto LaunchF = Function::Create(LaunchT, Function::InternalLinkage,
      "legion_task_launcher_create", &TheModule);

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = TmpB.CreateAlloca(LauncherT, nullptr, "task_launcher.alloca");
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
  auto ExecF = Function::Create(ExecT, Function::InternalLinkage,
      "legion_task_launcher_execute", &TheModule);
  
  auto FutureRV = Builder_.CreateCall(ExecF, ExecArgVs, "launcher_exec");
  auto FutureA = TmpB.CreateAlloca(FutureT, nullptr, "future.alloca");
  store(FutureRV, FutureA);

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");

  auto VoidT = llvmType<void>(TheContext_);

  auto DestroyT = FunctionType::get(VoidT, LauncherRT, false);
  auto DestroyF = Function::Create(DestroyT, Function::InternalLinkage,
      "legion_task_launcher_destroy", &TheModule);

  Builder_.CreateCall(DestroyF, LauncherRV, "launcher_destroy");
  
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  
 
  TheModule.print(outs(), nullptr); outs()<<"\n"; 
  abort();

}

}
