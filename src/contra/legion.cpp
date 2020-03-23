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

Type* reduceStruct(StructType * StructT, LLVMContext & TheContext, const Module &TheModule) {
  auto NumElem = StructT->getNumElements();
  auto ElementTs = StructT->elements();
  if (NumElem == 1) return ElementTs[0];
  auto DL = std::make_unique<DataLayout>(&TheModule);
  auto BitWidth = DL->getTypeAllocSizeInBits(StructT);
  return IntegerType::get(TheContext, BitWidth);
}
 
Value* castStruct(Value* V, BasicBlock * TheBlock, LLVMContext & TheContext,
    const Module &TheModule)
{
  auto T = V->getType();
  if (auto StrucT = dyn_cast<StructType>(T)) {
    auto NewT = reduceStruct(StrucT, TheContext, TheModule);
    std::string Str = StrucT->hasName() ? StrucT->getName().str()+".cast" : "casttmp";
    auto Cast = CastInst::Create(CastInst::BitCast, V, NewT, Str, TheBlock);
    return Cast;
  }
  else {
    return V;
  }
}

void castStructs(std::vector<Value*> & Vs, BasicBlock * TheBlock,
    LLVMContext & TheContext, const Module &TheModule )
{ for (auto & V : Vs ) V = castStruct(V, TheBlock, TheContext, TheModule); }

//==============================================================================
// Create the function wrapper
//==============================================================================
Function* LegionTasker::wrap(Module &TheModule, const std::string & Name,
    Function* TaskF) const
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
  auto TaskPtrT = TaskT->getPointerTo();
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
  auto ContextA = TmpB.CreateAlloca(ContextT, nullptr, "context.alloca");

  auto RuntimeT = createOpaqueType("legion_runtime_t", TheContext_);
  auto RuntimeA = TmpB.CreateAlloca(RuntimeT, nullptr, "runtime.alloca");

  // args
  std::vector<Value*> PreambleArgVs = { DataV, DataLenV, ProcIdV,
    TaskA, RegionsA, NumRegionsA, RuntimeA, ContextA };
  //castStructs(PreambleArgVs, Builder_.GetInsertBlock(), TheContext_, TheModule);
  auto PreambleArgTs = llvmTypes(PreambleArgVs);
  
  auto PreambleT = FunctionType::get(VoidT, PreambleArgTs, false);
  auto PreambleF = Function::Create(PreambleT, Function::InternalLinkage,
      "legion_task_preamble", &TheModule);
  
  Builder_.CreateCall(PreambleF, PreambleArgVs, "preamble");

  //----------------------------------------------------------------------------
  // extrat user variables
  std::vector<Value*> TaskArgVs;
  
  //----------------------------------------------------------------------------
  // call users actual function
  Builder_.CreateCall(TaskF, TaskArgVs, "calltmp");

  //----------------------------------------------------------------------------
  // Postable
 
  // temporaries
  auto RuntimeV = Builder_.CreateLoad(RuntimeT, RuntimeA, "runtime");
  auto ContextV = Builder_.CreateLoad(ContextT, ContextA, "context");

  auto RetvalV = Constant::getNullValue(VoidPtrT);
  auto RetsizeV = llvmValue<std::size_t>(TheContext_, 0);

  // args
  std::vector<Value*> PostambleArgVs = { RuntimeV, ContextV, RetvalV, RetsizeV };
  //castStructs(PostambleArgVs, Builder_.GetInsertBlock(), TheContext_, TheModule);
  std::vector<Type*> PostambleArgTs = llvmTypes(PostambleArgVs);

  // call
  auto PostambleT = FunctionType::get(VoidT, PostambleArgTs, false);
  auto PostambleF = Function::Create(PostambleT, Function::InternalLinkage,
      "legion_task_postamble", &TheModule);
  
  Builder_.CreateCall(PostambleF, PostambleArgVs, "preamble");
  
  //----------------------------------------------------------------------------
  // function retuns void
  Builder_.CreateRetVoid();

  return WrapperF;
}

//==============================================================================
// Preregister tasks
//==============================================================================
void LegionTasker::preregister(llvm::Module &TheModule, const std::string & Name,
    const TaskInfo & Task ) const
{
  // get current insertion point
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto TmpB = CodeGen::createBuilder(TheFunction);

  //----------------------------------------------------------------------------
  // execution_constraint_set
  auto ExecSetT = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  
  auto ExecT = FunctionType::get(ExecSetT, false);
  auto ExecF = Function::Create(ExecT, Function::InternalLinkage,
      "legion_execution_constraint_set_create", &TheModule);
  
  Value* ExecSetV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecSetA = TmpB.CreateAlloca(ExecSetT, nullptr, "execution_constraint.alloca");
  Builder_.CreateStore(ExecSetV, ExecSetA);
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto VoidT = llvmType<void>(TheContext_);
  auto ProcIdV = llvmValue<legion_processor_kind_t>(TheContext_, LOC_PROC);  
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  //castStructs(AddExecArgVs, Builder_.GetInsertBlock(), TheContext_, TheModule);
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidT, AddExecArgTs, false);
  auto AddExecF = Function::Create(AddExecT, Function::InternalLinkage,
      "legion_execution_constraint_set_add_processor_constraint", &TheModule);

  Builder_.CreateCall(AddExecF, AddExecArgVs, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  auto LayoutSetT = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  auto LayoutT = FunctionType::get(LayoutSetT, false);
  auto LayoutF = Function::Create(LayoutT, Function::InternalLinkage,
      "legion_task_layout_constraint_set_create", &TheModule);
  
  Value* LayoutSetV = Builder_.CreateCall(LayoutF, None, "layout_constraint");
  
  auto LayoutSetA = TmpB.CreateAlloca(LayoutSetT, nullptr, "layout_constraint.alloca");
  Builder_.CreateStore(LayoutSetV, LayoutSetA);
  
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
  auto TaskIdT = llvmType<legion_task_id_t>(TheContext_);
  auto TaskIdVariantT = llvmType<legion_variant_id_t>(TheContext_);
  auto SizeT = llvmType<std::size_t>(TheContext_);
  //auto FunT = Task.getFunction()->getFunctionType();
  //auto FunPtrT = FunT->getPointerTo();
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = Function::Create(TaskT, Function::InternalLinkage, Task.getName(), &TheModule);
  
  Value* TaskIdV = llvmValue<legion_task_id_t>(TheContext_, Task.getId());
  auto TaskIdVariantV = llvmValue<legion_variant_id_t>(TheContext_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  ExecSetV = Builder_.CreateLoad(ExecSetT, ExecSetA, "execution_constraints");
  LayoutSetV = Builder_.CreateLoad(LayoutSetT, LayoutSetA, "layout_constraints");
 

  auto I32T = llvmType<int>(TheContext_);
  auto Cast = CastInst::Create(CastInst::BitCast, TaskConfigA,
      I32T->getPointerTo(), "", Builder_.GetInsertBlock());
  auto OptionsV = Builder_.CreateLoad(I32T, Cast, "options");
  //auto OptionsV = Builder_.CreateLoad(TaskConfigT, TaskConfigA, "options");

  //auto TheBlock = Builder_.GetInsertBlock();
  //auto TaskInt = Task.getAddress();
  //auto TaskIntV = llvmValue<intptr_t>(TheContext_, TaskInt);
  //auto TaskPtr = CastInst::Create(Instruction::IntToPtr, TaskIntV, FunPtrT,
  //    "fptr", TheBlock);

  auto UserDataV = Constant::getNullValue(VoidPtrT);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, ExecSetV, LayoutSetV, OptionsV, TaskF, UserDataV, UserLenV };
  //castStructs(PreArgVs, Builder_.GetInsertBlock(), TheContext_, TheModule);
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
void LegionTasker::set_top(llvm::Module &TheModule, int TaskId ) const
{

  auto VoidT = llvmType<void>(TheContext_);
  auto TaskIdT = llvmType<legion_task_id_t>(TheContext_);
  
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
llvm::Value* LegionTasker::start(llvm::Module &TheModule, int Argc, char ** Argv) const
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

}
