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
Type * createOpaqueType(const std::string & Name, LLVMContext & TheContext)
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
Type * createTaskConfigOptionsType(const std::string & Name, LLVMContext & TheContext)
{
  auto OptionsType = StructType::create( TheContext, Name );
  auto BoolType = llvmType<bool>(TheContext);

  std::vector<Type*> members(4, BoolType);
  OptionsType->setBody( members );

  return OptionsType;
}

//==============================================================================
// Create the function wrapper
//==============================================================================
Function* LegionTasker::wrap(Module &TheModule, const std::string & Name,
    Function* TaskF) const
{
  auto VoidPtrType = llvmType<void*>(TheContext_);
  auto VoidType = Type::getVoidTy(TheContext_);
  auto SizeType = llvmType<std::size_t>(TheContext_);
  auto RealmIdType = llvmType<realm_id_t>(TheContext_);

  //----------------------------------------------------------------------------
  // Create task wrapper
  std::string TaskName = "__" + Name + "_task__";
  std::vector<Type *> WrapperArgTypes =
    {VoidPtrType, SizeType, VoidPtrType, SizeType, RealmIdType};
  
  auto WrapperT = FunctionType::get(VoidType, WrapperArgTypes, false);
  auto WrapperF = Function::Create(WrapperT, Function::InternalLinkage,
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
  std::vector<Value*> WrapperArgsV;
  WrapperArgsV.reserve(WrapperArgTypes.size());

  auto TmpB = CodeGen::createBuilder(WrapperF);

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgType = WrapperArgTypes[ArgIdx];
    // Create an alloca for this variable.
    auto Alloca = TmpB.CreateAlloca(ArgType, nullptr, Arg.getName());
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgsV.emplace_back(Alloca);
    ArgIdx++;
  }

  // loads
  auto DataV = Builder_.CreateLoad(VoidPtrType, WrapperArgsV[0], "data");
  auto DataLenV = Builder_.CreateLoad(SizeType, WrapperArgsV[1], "datalen");
  //auto UserDataV = Builder_.CreateLoad(VoidPtrType, WrapperArgsV[2], "userdata");
  //auto UserLenV = Builder_.CreateLoad(SizeType, WrapperArgsV[3], "userlen");
  auto ProcIdV = Builder_.CreateLoad(RealmIdType, WrapperArgsV[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto OpaqueType = createOpaqueType("legion_opaque_t", TheContext_);
  auto OpaquePtrType = PointerType::get(OpaqueType, 0);
  
  auto TaskAlloca = TmpB.CreateAlloca(OpaqueType, nullptr, "task");
  
  auto RegionsAlloca = TmpB.CreateAlloca(OpaquePtrType, nullptr, "regions");
  auto NullV = Constant::getNullValue(OpaquePtrType);
  Builder_.CreateStore(NullV, RegionsAlloca);

  auto NumType = llvmType<std::uint32_t>(TheContext_); 
  auto NumRegionsAlloca = TmpB.CreateAlloca(NumType, nullptr, "num_regions");
  auto ZeroV = ConstantInt::get(TheContext_, APInt(32, 0 /* len for now */, false));  
  Builder_.CreateStore(ZeroV, NumRegionsAlloca);
  
  auto ContextAlloca = TmpB.CreateAlloca(OpaqueType, nullptr, "ctx");
  auto RuntimeAlloca = TmpB.CreateAlloca(OpaqueType, nullptr, "runtime");

  // args
  std::vector<Value*> PreambleArgsV = { DataV, DataLenV, ProcIdV,
    TaskAlloca, RegionsAlloca, NumRegionsAlloca, RuntimeAlloca, ContextAlloca };

  std::vector<Type*> PreambleArgTypes;
  PreambleArgTypes.reserve(PreambleArgsV.size());
  for (auto & Arg : PreambleArgsV) PreambleArgTypes.emplace_back( Arg->getType() );
  
  auto PreambleT = FunctionType::get(VoidType, PreambleArgTypes, false);
  auto PreambleF = Function::Create(PreambleT, Function::ExternalLinkage,
      "legion_task_preamble", &TheModule);
  
  Builder_.CreateCall(PreambleF, PreambleArgsV, "preamble");

  //----------------------------------------------------------------------------
  // extrat user variables
  std::vector<Value*> TaskArgsV;
  
  //----------------------------------------------------------------------------
  // call users actual function
  Builder_.CreateCall(TaskF, TaskArgsV, "calltmp");

  //----------------------------------------------------------------------------
  // Postable
 
  // temporaries
  auto RuntimeV = Builder_.CreateLoad(OpaqueType, RuntimeAlloca, "runtime");
  auto ContextV = Builder_.CreateLoad(OpaqueType, ContextAlloca, "ctx");

  auto RetvalV = Constant::getNullValue(VoidPtrType);
  auto RetsizeV = llvmValue<std::size_t>(TheContext_, 0);

  // args
  std::vector<Value*> PostambleArgsV = { RuntimeV, ContextV, RetvalV, RetsizeV };
  
  std::vector<Type*> PostambleArgTypes;
  PostambleArgTypes.reserve(PostambleArgsV.size());
  for (auto & Arg : PostambleArgsV) PostambleArgTypes.emplace_back( Arg->getType() );

  // call
  auto PostambleT = FunctionType::get(VoidType, PostambleArgTypes, false);
  auto PostambleF = Function::Create(PostambleT, Function::ExternalLinkage,
      "legion_task_postamble", &TheModule);
  
  Builder_.CreateCall(PostambleF, PostambleArgsV, "preamble");
  
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
  auto OpaqueT = createOpaqueType("legion_opaque_t", TheContext_);
  
  auto ExecT = FunctionType::get(OpaqueT, false);
  auto ExecF = Function::Create(ExecT, Function::ExternalLinkage,
      "legion_execution_constraint_set_create", &TheModule);
  
  Value* ExecV = Builder_.CreateCall(ExecF, None, "execution_constraint");
  
  auto ExecAlloca = TmpB.CreateAlloca(OpaqueT, nullptr, "legion_execution_constraint_set_t");
  Builder_.CreateStore(ExecV, ExecAlloca);

  //----------------------------------------------------------------------------
  // add constraint
  //
  auto ProcIdT = llvmType<legion_processor_kind_t>(TheContext_);
  auto ProcIdV = llvmValue<legion_processor_kind_t>(TheContext_, LOC_PROC);  
  
  std::vector<Type*> AddExecArgTs = {OpaqueT, ProcIdT};
  auto AddExecT = FunctionType::get(OpaqueT, AddExecArgTs, false);
  auto AddExecF = Function::Create(AddExecT, Function::ExternalLinkage,
      "legion_execution_constraint_set_add_processor_constraint", &TheModule);

  std::vector<Value*> AddExecArgsV = {ExecV, ProcIdV};
  Builder_.CreateCall(AddExecF, AddExecArgsV, "add_processor_constraint");

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
  
  auto LayoutT = FunctionType::get(OpaqueT, false);
  auto LayoutF = Function::Create(LayoutT, Function::ExternalLinkage,
      "legion_task_layout_constraint_set_create", &TheModule);
  
  Value* LayoutV = Builder_.CreateCall(LayoutF, None, "execution_constraint");
  
  auto LayoutAlloca = TmpB.CreateAlloca(OpaqueT, nullptr, "task_layout_constraint");
  Builder_.CreateStore(LayoutV, LayoutAlloca);
  
  //----------------------------------------------------------------------------
  // options
  auto TaskConfigT = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  auto TaskConfigAlloca = TmpB.CreateAlloca(TaskConfigT, nullptr, "options");
  auto ZeroV = ConstantInt::get(TheContext_, APInt(8, 0, false));  
  Builder_.CreateMemSet(TaskConfigAlloca, ZeroV, 4, 1); 
  
  //----------------------------------------------------------------------------
  // registration
  
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto TaskIdT = llvmType<legion_task_id_t>(TheContext_);
  auto TaskIdVariantT = llvmType<legion_variant_id_t>(TheContext_);
  auto SizeT = llvmType<std::size_t>(TheContext_);
  auto FunPtrT = Task.Func->getFunctionType()->getPointerTo();
  
  std::vector<Type*> PreArgTs = {TaskIdT, TaskIdVariantT, VoidPtrT,
    VoidPtrT, OpaqueT, OpaqueT, TaskConfigT, FunPtrT, VoidPtrT,
    SizeT};
  auto PreRetT = TaskIdVariantT;
  auto PreT = FunctionType::get(PreRetT, PreArgTs, false);
  auto PreF = Function::Create(PreT, Function::ExternalLinkage,
      "legion_runtime_preregister_task_variant_fnptr", &TheModule);

  Value* TaskIdV = llvmValue<legion_task_id_t>(TheContext_, Task.Id);
  auto TaskIdVariantV = llvmValue<legion_variant_id_t>(TheContext_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  ExecV = Builder_.CreateLoad(ExecT, ExecAlloca, "execution_constraints");
  LayoutV = Builder_.CreateLoad(LayoutT, LayoutAlloca, "layout_constraints");
  
  auto OptionsV = Builder_.CreateLoad(TaskConfigT, TaskConfigAlloca, "options");

  auto TheBlock = Builder_.GetInsertBlock();
  auto TaskInt = reinterpret_cast<intptr_t>(Task.Func);
  auto TaskIntV = llvmValue<intptr_t>(TheContext_, TaskInt);
  auto TaskPtr = CastInst::Create(Instruction::IntToPtr, TaskIntV, FunPtrT,
      "fptr", TheBlock);

  auto UserDataV = Constant::getNullValue(VoidPtrT);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, ExecV, LayoutV, OptionsV, TaskPtr, UserDataV, UserLenV };

  TaskIdV = Builder_.CreateCall(PreF, PreArgVs, "task_variant_id");

}
  
//==============================================================================
// Set top level task
//==============================================================================
void LegionTasker::set_top(llvm::Module &TheModule, int TaskId ) const
{

  auto VoidT = Type::getVoidTy(TheContext_);
  auto TaskIdT = llvmType<legion_task_id_t>(TheContext_);
  std::vector<Type*> SetArgTs = { TaskIdT };
  auto SetT = FunctionType::get(VoidT, SetArgTs, false);
  auto SetF = Function::Create(SetT, Function::ExternalLinkage,
      "legion_runtime_set_top_level_task_id", &TheModule);

  auto TaskIdV = llvmValue<legion_task_id_t>(TheContext_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
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
