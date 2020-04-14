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
LegionTasker::LegionTasker(IRBuilder<> & TheBuilder, LLVMContext & TheContext) :
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
  FutureIdType_ = llvmType<unsigned>(TheContext_);
  CoordType_ = llvmType<legion_coord_t>(TheContext_);
  Point1dType_ = ArrayType::get(CoordType_, 1);

  TaskType_ = createOpaqueType("legion_task_t", TheContext_);
  RegionType_ = createOpaqueType("legion_physical_region_t", TheContext_);
  ContextType_ = createOpaqueType("legion_context_t", TheContext_);
  RuntimeType_ = createOpaqueType("legion_runtime_t", TheContext_);
  ExecSetType_ = createOpaqueType("legion_execution_constraint_set_t", TheContext_);
  LayoutSetType_ = createOpaqueType("legion_task_layout_constraint_set_t", TheContext_);
  PredicateType_ = createOpaqueType("legion_predicate_t", TheContext_);
  TaskLauncherType_ = createOpaqueType("legion_task_launcher_t", TheContext_);
  IndexLauncherType_ = createOpaqueType("legion_index_launcher_t", TheContext_);
  FutureType_ = createOpaqueType("legion_future_t", TheContext_);
  TaskConfigType_ = createTaskConfigOptionsType("task_config_options_t", TheContext_);
  TaskArgsType_ = createTaskArgumentsType("legion_task_argument_t", TheContext_);
  DomainPointType_ = createDomainPointType("legion_domain_point_t", TheContext_);
  Rect1dType_ = createRect1dType("legion_rect_1d_t", TheContext_);
  DomainRectType_ = createDomainRectType("legion_domain_t", TheContext_);
  ArgMapType_ = createOpaqueType("legion_argument_map_t", TheContext_);
  FutureMapType_ = createOpaqueType("legion_future_map_t", TheContext_);
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
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createDomainPointType(
    const std::string & Name, LLVMContext & TheContext)
{
  auto ArrayT = ArrayType::get(CoordType_, MAX_POINT_DIM); 
  std::vector<Type*> members = { Int32Type_, ArrayT };
  auto NewType = StructType::create( TheContext, members, Name );
  return NewType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createRect1dType(
    const std::string & Name, LLVMContext & TheContext)
{
  std::vector<Type*> members = { Point1dType_, Point1dType_ };
  auto NewType = StructType::create( TheContext, members, Name );
  return NewType;
}
  

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createDomainRectType(
    const std::string & Name, LLVMContext & TheContext)
{
  auto ArrayT = ArrayType::get(CoordType_, 2*LEGION_MAX_DIM); 
  std::vector<Type*> members = { RealmIdType_, Int32Type_, ArrayT };
  auto NewType = StructType::create( TheContext, members, Name );
  return NewType;
}
  
//==============================================================================
// Create a true predicate
//==============================================================================
AllocaInst* LegionTasker::createPredicateTrue(Module &TheModule)
{
  auto PredicateRT = reduceStruct(PredicateType_, TheModule);
  
  auto PredicateTrueT = FunctionType::get(PredicateRT, false);
  auto PredicateTrueF = TheModule.getOrInsertFunction(
      "legion_predicate_true", PredicateTrueT);
  
  Value* PredicateRV = Builder_.CreateCall(PredicateTrueF, None, "pred_true");
  
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto PredicateA = createEntryBlockAlloca(TheFunction, PredicateType_, "predicate.alloca");
  store(PredicateRV, PredicateA);

  return PredicateA;
}

//==============================================================================
// Codegen the global arguments
//==============================================================================
AllocaInst* LegionTasker::createGlobalArguments(
    Module &TheModule,
    const std::vector<Value*> & ArgVs,
    const std::vector<Value*> & ArgSizes,
    std::vector<unsigned> & FutureArgId
    )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  auto TaskArgsA = createEntryBlockAlloca(TheFunction, TaskArgsType_, "args.alloca");

  //----------------------------------------------------------------------------
  // Identify futures
  
  auto NumArgs = ArgVs.size();
  
  std::vector<char> ArgIsFuture(NumArgs);
  std::vector<Constant*> ArgIsFutureC(NumArgs);
  std::vector<unsigned> ValueArgId;

  unsigned NumFutureArgs = 0;
  for (unsigned i=0; i<NumArgs; i++) {
    ArgIsFuture[i] = (ArgVs[i]->getType() == FutureType_);
    NumFutureArgs += ArgIsFuture[i];
    if (ArgIsFuture[i]) FutureArgId.emplace_back(i);
    else ValueArgId.emplace_back(i);
    ArgIsFutureC[i] = llvmValue(TheContext_, BoolType_, ArgIsFuture[i]);
  }

  //----------------------------------------------------------------------------
  // First count sizes

  auto ArgSizeGEP = accessStructMember(TaskArgsA, 1, "arglen");
  auto ArgSizeT = TaskArgsType_->getElementType(1);
  // add 1 byte for each argument first
  Builder_.CreateStore( llvmValue(TheContext_, ArgSizeT, NumArgs), ArgSizeGEP );

  for (auto i : ValueArgId) {
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
  // create an array with booleans identifying argyment type
  
  auto ArrayT = ArrayType::get(BoolType_, NumArgs);
  auto ArrayC = ConstantArray::get(ArrayT, ArgIsFutureC);
  auto GVStr = new GlobalVariable(TheModule, ArrayT, true, GlobalValue::InternalLinkage, ArrayC);
  auto ZeroC = Constant::getNullValue(Int32Type_);
  auto ArrayGEP = ConstantExpr::getGetElementPtr(nullptr, GVStr, ZeroC, true);
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  Builder_.CreateMemCpy(ArgDataPtrV, 1, ArrayGEP, 1, llvmValue<size_t>(TheContext_, NumArgs)); 
 
  //----------------------------------------------------------------------------
  // Copy args
  
  // add 1 byte for each argument first
  Builder_.CreateStore( llvmValue(TheContext_, ArgSizeT, NumArgs), ArgSizeGEP );
  
  for (auto i : ValueArgId) {
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

  return TaskArgsA; 
}

//==============================================================================
// Codegen the global future arguments
//==============================================================================
void LegionTasker::createGlobalFutures(
    llvm::Module & TheModule,
    Value* LauncherA,
    const std::vector<Value*> & ArgVs,
    const std::vector<Value*> & ArgSizes,
    const std::vector<unsigned> & FutureArgId,
    bool IsIndex )
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  auto FutureRT = reduceStruct(FutureType_, TheModule);

  StructType* LauncherT = IsIndex ? IndexLauncherType_ : TaskLauncherType_;
  auto LauncherRT = reduceStruct(LauncherT, TheModule);

  std::vector<Type*> AddFutureArgTs = {LauncherRT, FutureRT};
  auto AddFutureT = FunctionType::get(VoidType_, AddFutureArgTs, false);

  FunctionCallee AddFutureF;
  if (IsIndex)
    AddFutureF = TheModule.getOrInsertFunction("legion_index_launcher_add_future", AddFutureT);
  else
    AddFutureF = TheModule.getOrInsertFunction("legion_task_launcher_add_future", AddFutureT);

  for (auto i : FutureArgId) {
    auto FutureA = createEntryBlockAlloca(TheFunction, FutureType_, "tmpalloca");
    Builder_.CreateStore( ArgVs[i], FutureA );
    auto FutureRV = load(FutureA, TheModule, "future");
    auto LauncherRV = load(LauncherA, TheModule, "task_launcher");
    std::vector<Value*> AddFutureArgVs = {LauncherRV, FutureRV};
    Builder_.CreateCall(AddFutureF, AddFutureArgVs);
  }
}

//==============================================================================
// Destroy an opaque type
//==============================================================================
AllocaInst* LegionTasker::createOpaqueType(Module& TheModule, StructType* OpaqueT,
    const std::string & FuncN, const std::string & Name)
{
  auto OpaqueRT = reduceStruct(OpaqueT, TheModule);
  
  auto ExecT = FunctionType::get(OpaqueRT, false);
  auto ExecF = TheModule.getOrInsertFunction(FuncN, ExecT);
  
  Value* OpaqueRV = Builder_.CreateCall(ExecF, None, Name);
  
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  auto OpaqueA = createEntryBlockAlloca(TheFunction, OpaqueT, Name);
  store(OpaqueRV, OpaqueA);
  return OpaqueA;
}
  

//==============================================================================
// Destroy an opaque type
//==============================================================================
void LegionTasker::destroyOpaqueType(Module& TheModule, Value* OpaqueA,
    const std::string & FuncN, const std::string & Name)
{
  auto OpaqueRV = load(OpaqueA, TheModule, Name);
  auto OpaqueRT = OpaqueRV->getType();

  auto DestroyOpaqueT = FunctionType::get(VoidType_, OpaqueRT, false);
  auto DestroyOpaqueF = TheModule.getOrInsertFunction(FuncN, DestroyOpaqueT);

  Builder_.CreateCall(DestroyOpaqueF, OpaqueRV);
}
  
//==============================================================================
// Destroy task arguments
//==============================================================================
void LegionTasker::destroyGlobalArguments(Module& TheModule, AllocaInst* TaskArgsA)
{
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  auto TmpA = Builder_.CreateAlloca(VoidPtrType_, nullptr); // not needed but InsertAtEnd doesnt work
  CallInst::CreateFree(ArgDataPtrV, TmpA);
  TmpA->eraseFromParent();
}


//==============================================================================
// create registration arguments
//==============================================================================
void LegionTasker::createRegistrationArguments(Module& TheModule,
    llvm::AllocaInst *& ExecSetA, llvm::AllocaInst *& LayoutSetA,
    llvm::AllocaInst *& TaskConfigA)
{
  // get current insertion point
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  //----------------------------------------------------------------------------
  // execution_constraint_set
  
  ExecSetA = createOpaqueType(TheModule, ExecSetType_,
      "legion_execution_constraint_set_create", "execution_constraint");
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto ProcIdV = llvmValue(TheContext_, ProcIdType_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  auto AddExecArgTs = llvmTypes(AddExecArgVs);
  auto AddExecT = FunctionType::get(VoidType_, AddExecArgTs, false);
  auto AddExecF = TheModule.getOrInsertFunction(
      "legion_execution_constraint_set_add_processor_constraint", AddExecT);

  Builder_.CreateCall(AddExecF, AddExecArgVs);

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  LayoutSetA = createOpaqueType(TheModule, LayoutSetType_,
      "legion_task_layout_constraint_set_create", "layout_constraint"); 
  
  //----------------------------------------------------------------------------
  // options
  TaskConfigA = createEntryBlockAlloca(TheFunction, TaskConfigType_, "options");
  auto BoolT = TaskConfigType_->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  Builder_.CreateMemSet(TaskConfigA, FalseV, 4, 1); 
}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(Module &TheModule,
    const std::string & TaskName, const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs, bool IsIndex)
{
  //----------------------------------------------------------------------------
  // Create task wrapper
  
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

  // create the context
  auto & LegionE = startTask();
  auto & ContextA = LegionE.ContextAlloca;
  auto & RuntimeA = LegionE.RuntimeAlloca;
  ContextA = createEntryBlockAlloca(WrapperF, ContextType_, "context.alloca");
  RuntimeA = createEntryBlockAlloca(WrapperF, RuntimeType_, "runtime.alloca");


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
 
  // args
  std::vector<Value*> PreambleArgVs = { DataV, DataLenV, ProcIdV,
    TaskA, RegionsA, NumRegionsA, ContextA, RuntimeA };
  auto PreambleArgTs = llvmTypes(PreambleArgVs);
  
  auto PreambleT = FunctionType::get(VoidType_, PreambleArgTs, false);
  auto PreambleF = TheModule.getOrInsertFunction("legion_task_preamble", PreambleT);
  
  Builder_.CreateCall(PreambleF, PreambleArgVs);
  
  //----------------------------------------------------------------------------
  // Get task args

  auto TaskRV = load(TaskA, TheModule, "task");
  auto TaskGetArgsT = FunctionType::get(VoidPtrType_, TaskRV->getType(), false);
  auto TaskGetArgsF = TheModule.getOrInsertFunction("legion_task_get_args", TaskGetArgsT);
  
  Value* TaskArgsV = Builder_.CreateCall(TaskGetArgsF, TaskRV, "args");
  
  auto TaskArgsA = createEntryBlockAlloca(WrapperF, VoidPtrType_, "args.alloca");
  Builder_.CreateStore(TaskArgsV, TaskArgsA);
  
  //----------------------------------------------------------------------------
  // Allocas for task arguments

  auto NumArgs = TaskArgTs.size();
  std::vector<AllocaInst*> TaskArgAs;
  std::vector<Value*> ArgSizes;
  for (unsigned i=0; i<NumArgs; ++i) {
    auto ArgT = TaskArgTs[i];
    auto ArgN = TaskArgNs[i] + ".alloca";
    auto ArgA = createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    TaskArgAs.emplace_back(ArgA);
    ArgSizes.emplace_back( getTypeSize<size_t>(Builder_, ArgT) );
  }
  
  //----------------------------------------------------------------------------
  // get user types

  auto ArrayT = ArrayType::get(BoolType_, NumArgs);
  auto ArrayA = createEntryBlockAlloca(WrapperF, ArrayT, "isfuture");
  auto ArgDataPtrV = Builder_.CreateLoad(VoidPtrType_, TaskArgsA, "args");
  Builder_.CreateMemCpy(ArrayA, 1, ArgDataPtrV, 1, llvmValue<size_t>(TheContext_, NumArgs)); 
  
  //----------------------------------------------------------------------------
  // unpack user variables
  
  auto OffsetT = SizeType_;
  auto OffsetA = createEntryBlockAlloca(WrapperF, OffsetT, "offset.alloca");
  Builder_.CreateStore( llvmValue(TheContext_, OffsetT, NumArgs), OffsetA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArrayV = Builder_.CreateLoad(ArrayT, ArrayA, "isfuture");
    auto ArgTypeV = Builder_.CreateExtractValue(ArrayV, i);
    auto OneV = llvmValue(TheContext_, ArgTypeV->getType(), 1);
    auto CondV = Builder_.CreateICmpNE(ArgTypeV, OneV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // offset 
    auto ArgGEP = offsetPointer(TaskArgsA, OffsetA, "args");
    // copy
    memCopy(ArgGEP, TaskArgAs[i], ArgSizes[i]);
    // increment
    increment(OffsetA, ArgSizes[i], "offset");
    // finish then
    ThenBB->getFirstNonPHI();
    Builder_.CreateBr(MergeBB);
    ThenBB = Builder_.GetInsertBlock();
    // Emit merge block.
    WrapperF->getBasicBlockList().push_back(MergeBB);
    Builder_.SetInsertPoint(MergeBB);
  }

  //----------------------------------------------------------------------------
  // unpack future variables
  
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto TaskRT = reduceStruct(TaskType_, TheModule);
  std::vector<Type*> GetFutureArgTs = {TaskRT, FutureIdType_};

  auto GetFutureT = FunctionType::get(FutureRT, GetFutureArgTs, false);
  auto GetFutureF = TheModule.getOrInsertFunction("legion_task_get_future", GetFutureT);
  
  auto FutureIndexT = FutureIdType_;
  auto FutureIndexA = createEntryBlockAlloca(WrapperF, FutureIndexT, "futureid.alloca");
  Builder_.CreateStore( Constant::getNullValue(FutureIndexT), FutureIndexA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(TheContext_, "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(TheContext_, "ifcont");
    // evaluate the condition
    auto ArrayV = Builder_.CreateLoad(ArrayT, ArrayA, "isfuture");
    auto ArgTypeV = Builder_.CreateExtractValue(ArrayV, i);
    auto OneV = llvmValue(TheContext_, ArgTypeV->getType(), 1);
    auto CondV = Builder_.CreateICmpEQ(ArgTypeV, OneV, "argtype");
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    Builder_.SetInsertPoint(ThenBB);
    // get future
    auto TaskRV = load(TaskA, TheModule, "task");
    auto FutureIndexV = Builder_.CreateLoad(FutureIndexT, FutureIndexA, "futureid");
    std::vector<Value*> GetFutureArgVs = {TaskRV, FutureIndexV};
    auto FutureRV = Builder_.CreateCall(GetFutureF, GetFutureArgVs, "get_future");
    auto FutureA = createEntryBlockAlloca(WrapperF, FutureType_, "future");
    store(FutureRV, FutureA);
    // unpack
    auto ArgA = TaskArgAs[i];
    auto ArgT = ArgA->getType()->getPointerElementType();
    // copy
    auto ArgV = loadFuture(TheModule, FutureA, ArgT, ArgSizes[i]);
    Builder_.CreateStore( ArgV, ArgA );
    // consume the future
    destroyFuture(TheModule, FutureA);
    // increment
    increment(FutureIndexA, llvmValue(TheContext_, FutureIndexT, 1), "futureid");
    // finish then
    ThenBB->getFirstNonPHI();
    Builder_.CreateBr(MergeBB);
    ThenBB = Builder_.GetInsertBlock();
    // Emit merge block.
    WrapperF->getBasicBlockList().push_back(MergeBB);
    Builder_.SetInsertPoint(MergeBB);
  }

  //----------------------------------------------------------------------------
  // If this is an index task
  
  AllocaInst* IndexA = nullptr;
  if (IsIndex) {
     
    auto TaskRV = load(TaskA, TheModule, "task");
    auto DomainPointA = createEntryBlockAlloca(WrapperF, DomainPointType_, "domain_point.alloca");

    std::vector<Value*> GetIndexArgVs = {DomainPointA, TaskRV};
    auto GetIndexArgTs = llvmTypes(GetIndexArgVs);
    auto GetIndexT = FunctionType::get(VoidType_, GetIndexArgTs, false);
    auto GetIndexF = TheModule.getFunction("legion_task_get_index_point");
    if (!GetIndexF) {
      GetIndexF = Function::Create(GetIndexT, Function::InternalLinkage,
          "legion_task_get_index_point", &TheModule);
      auto Arg = GetIndexF->arg_begin();
      Arg->addAttr(Attribute::StructRet);
    }
 
    Builder_.CreateCall(GetIndexF, GetIndexArgVs);

    auto PointDataGEP = accessStructMember(DomainPointA, 1, "point_data");
    auto PointDataT = DomainPointType_->getElementType(1);
    auto PointDataV = Builder_.CreateLoad(PointDataT, PointDataGEP);
    auto IndexV = Builder_.CreateExtractValue(PointDataV, 0);

    IndexA = createEntryBlockAlloca(WrapperF, llvmType<int_t>(TheContext_), "index");
    Builder_.CreateStore( IndexV, IndexA );
  }

  //----------------------------------------------------------------------------
  // Function body
  return {WrapperF, TaskArgAs, IndexA}; 

}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(Module &TheModule,
    const std::string & Name, Function* TaskF)
{

  std::string TaskName = "__" + Name + "_task__";
  
  std::vector<Type*> TaskArgTs;
  std::vector<std::string> TaskArgNs;
 
  for (auto & Arg : TaskF->args()) {
    auto ArgT = Arg.getType();
    auto ArgN = Arg.getName().str();
    TaskArgTs.emplace_back(ArgT);
    TaskArgNs.emplace_back(ArgN);
  }

  return taskPreamble(TheModule, TaskName, TaskArgNs, TaskArgTs);
}
  
//==============================================================================
// Create the function wrapper
//==============================================================================
void LegionTasker::taskPostamble(Module &TheModule, Value* ResultV)
{

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
  
  // temporaries
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");
  auto ContextV = load(ContextA, TheModule, "context");
  

  // args
  std::vector<Value*> PostambleArgVs = { RuntimeV, ContextV, RetvalV, RetsizeV };
  sanitize(PostambleArgVs, TheModule);
  std::vector<Type*> PostambleArgTs = llvmTypes(PostambleArgVs);

  // call
  auto PostambleT = FunctionType::get(VoidType_, PostambleArgTs, false);
  auto PostambleF = TheModule.getOrInsertFunction("legion_task_postamble", PostambleT);
  
  Builder_.CreateCall(PostambleF, PostambleArgVs);
  
  //----------------------------------------------------------------------------
  // Free memory
  if (ResultV) {
    auto RetvalT = RetvalV->getType();
    RetvalV = Builder_.CreateLoad(RetvalT, RetvalA);
    auto TmpA = Builder_.CreateAlloca(VoidPtrType_, nullptr); // not needed but InsertAtEnd doesnt work
    CallInst::CreateFree(RetvalV, TmpA);
    TmpA->eraseFromParent();
  }

  
  finishTask();
}

//==============================================================================
// Postregister tasks
//==============================================================================
void LegionTasker::postregisterTask(Module &TheModule, const std::string & Name,
    const TaskInfo & Task )
{

  //----------------------------------------------------------------------------
  // arguments
  
  llvm::AllocaInst *ExecSetA, *LayoutSetA, *TaskConfigA;
  createRegistrationArguments(TheModule, ExecSetA, LayoutSetA, TaskConfigA);
  
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(TheContext_, TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(TheContext_, TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(TheContext_, 0);
  
  auto BoolT = TaskConfigType_->getElementType(0);
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
void LegionTasker::preregisterTask(Module &TheModule, const std::string & Name,
    const TaskInfo & Task )
{

  //----------------------------------------------------------------------------
  // arguments
  
  llvm::AllocaInst *ExecSetA, *LayoutSetA, *TaskConfigA;
  createRegistrationArguments(TheModule, ExecSetA, LayoutSetA, TaskConfigA);
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(TheContext_, TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(TheContext_, TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(TheContext_, TheModule, Name + " task");
  auto VariantNameV = llvmString(TheContext_, TheModule, Name + " variant");

  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
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
void LegionTasker::setTopLevelTask(Module &TheModule, int TaskId )
{

  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
  auto SetArgTs = llvmTypes(SetArgVs);

  auto SetT = FunctionType::get(VoidType_, SetArgTs, false);
  auto SetF = TheModule.getOrInsertFunction(
      "legion_runtime_set_top_level_task_id", SetT);

  Builder_.CreateCall(SetF, SetArgVs);
}
  
//==============================================================================
// start runtime
//==============================================================================
Value* LegionTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
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
Value* LegionTasker::launch(Module &TheModule, const std::string & Name,
    int TaskId, const std::vector<Value*> & ArgVs,
    const std::vector<Value*> & ArgSizes)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  //----------------------------------------------------------------------------
  // Global arguments
  std::vector<unsigned> FutureArgId;
  auto TaskArgsA = createGlobalArguments(TheModule, ArgVs, ArgSizes, FutureArgId);
  
 
  //----------------------------------------------------------------------------
  // Predicate
  auto PredicateA = createPredicateTrue(TheModule);
  
  //----------------------------------------------------------------------------
  // Launch
 
  auto MapperIdV = llvmValue(TheContext_, MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(TheContext_, MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  auto ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");

  auto LauncherRT = reduceStruct(TaskLauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, ArgDataPtrV, ArgSizeV, 
    PredicateV, MapperIdV, MappingTagIdV};
  auto LaunchArgTs = llvmTypes(LaunchArgVs);

  auto LaunchT = FunctionType::get(LauncherRT, LaunchArgTs, false);
  auto LaunchF = TheModule.getOrInsertFunction("legion_task_launcher_create", LaunchT);

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = createEntryBlockAlloca(TheFunction, TaskLauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Add futures
  createGlobalFutures(TheModule, LauncherA, ArgVs, ArgSizes, FutureArgId, false );
  
  //----------------------------------------------------------------------------
  // Execute
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextA, TheModule, "context");
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

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
  
  destroyOpaqueType(TheModule, LauncherA, "legion_task_launcher_destroy", "task_launcher");
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  destroyGlobalArguments(TheModule, TaskArgsA);

  return Builder_.CreateLoad(FutureType_, FutureA);
}

//==============================================================================
// Launch an index task
//==============================================================================
Value* LegionTasker::launch(Module &TheModule, const std::string & Name,
    int TaskId, const std::vector<Value*> & ArgVs,
    const std::vector<Value*> & ArgSizes, Value* StartA, Value* EndA)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  //----------------------------------------------------------------------------
  // Global arguments
  std::vector<unsigned> FutureArgId;
  auto TaskArgsA = createGlobalArguments(TheModule, ArgVs, ArgSizes, FutureArgId);

  //----------------------------------------------------------------------------
  // Predicate
  auto PredicateA = createPredicateTrue(TheModule);
  
  //----------------------------------------------------------------------------
  // Create domain  
  
  auto DomainLoA = createEntryBlockAlloca(TheFunction, Point1dType_, "lo");
  auto DomainHiA = createEntryBlockAlloca(TheFunction, Point1dType_, "hi");

  auto StartT = StartA->getType()->getPointerElementType();
  Value* StartV = Builder_.CreateLoad(StartT, StartA);
  auto DomainLoV = Builder_.CreateLoad(Point1dType_, DomainLoA); 
  auto ZeroC = Constant::getNullValue(Int32Type_);
  std::vector<Value*> IndicesC = {ZeroC,  ZeroC};
  auto DomainLoGEP = Builder_.CreateGEP(DomainLoA, IndicesC, "lo");
	Builder_.CreateStore(StartV, DomainLoGEP);
  
  auto EndT = EndA->getType()->getPointerElementType();
  Value* EndV = Builder_.CreateLoad(EndT, EndA);
  auto DomainHiV = Builder_.CreateLoad(Point1dType_, DomainHiA); 
  auto DomainHiGEP = Builder_.CreateGEP(DomainHiA, IndicesC, "hi");
	Builder_.CreateStore(EndV, DomainHiGEP);

  auto LaunchBoundA = createEntryBlockAlloca(TheFunction, Rect1dType_, "launch_bound");
  
  DomainLoV = Builder_.CreateLoad(Point1dType_, DomainLoA);
  storeStructMember(DomainLoV, LaunchBoundA, 0);
  
  DomainHiV = Builder_.CreateLoad(Point1dType_, DomainHiA);
  storeStructMember(DomainHiV, LaunchBoundA, 1);

  auto DomainLoRV = load(DomainLoA, TheModule, "lo");
  auto DomainHiRV = load(DomainHiA, TheModule, "hi");
  auto DomainRectA = createEntryBlockAlloca(TheFunction, DomainRectType_, "domain");
  std::vector<Value*> DomainFromArgVs = { DomainRectA, DomainLoRV, DomainHiRV };
  auto DomainFromArgTs = llvmTypes(DomainFromArgVs);

  auto DomainFromT = FunctionType::get(VoidType_, DomainFromArgTs, false);
  auto DomainFromF = TheModule.getFunction("legion_domain_from_rect_1d");
  if (!DomainFromF) {
    DomainFromF = Function::Create(DomainFromT, Function::InternalLinkage,
        "legion_domain_from_rect_1d", &TheModule);
    auto Arg = DomainFromF->arg_begin();
    Arg->addAttr(Attribute::StructRet);
  }
  
  Builder_.CreateCall(DomainFromF, DomainFromArgVs);
  
  //----------------------------------------------------------------------------
  // argument map 
  
  auto ArgMapA = createOpaqueType(TheModule, ArgMapType_,
      "legion_argument_map_create", "arg_map");
  
  //----------------------------------------------------------------------------
  // Launch
 
  auto MapperIdV = llvmValue(TheContext_, MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(TheContext_, MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(TheContext_, TaskIdType_, TaskId);
  auto MustV = Constant::getNullValue(BoolType_);
  auto ArgMapV = load(ArgMapA, TheModule, "arg_map");
  
  auto ArgDataPtrV = loadStructMember(TaskArgsA, 0, "args");
  auto ArgSizeV = loadStructMember(TaskArgsA, 1, "arglen");

  auto LauncherRT = reduceStruct(IndexLauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, DomainRectA, ArgDataPtrV, ArgSizeV, 
    ArgMapV, PredicateV, MustV, MapperIdV, MappingTagIdV};
  auto LaunchArgTs = llvmTypes(LaunchArgVs);

  auto LaunchT = FunctionType::get(LauncherRT, LaunchArgTs, false);
  auto LaunchF = TheModule.getFunction("legion_index_launcher_create");
  if (!LaunchF) {
    LaunchF = Function::Create(LaunchT, Function::InternalLinkage,
        "legion_index_launcher_create", &TheModule);
    auto Arg = LaunchF->arg_begin();
    ++Arg;
    Arg->addAttr(Attribute::ByVal);
  }

  Value* LauncherRV = Builder_.CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = createEntryBlockAlloca(TheFunction, IndexLauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Add futures
  createGlobalFutures(TheModule, LauncherA, ArgVs, ArgSizes, FutureArgId, true );
 
  
  //----------------------------------------------------------------------------
  // Execute
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextA, TheModule, "context");
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  // args
  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };
  auto ExecArgTs = llvmTypes(ExecArgVs);
  auto FutureMapRT = reduceStruct(FutureMapType_, TheModule);

  auto ExecT = FunctionType::get(FutureMapRT, ExecArgTs, false);
  auto ExecF = TheModule.getOrInsertFunction("legion_index_launcher_execute", ExecT);
  
  Value* FutureMapRV = Builder_.CreateCall(ExecF, ExecArgVs, "launcher_exec");
  auto FutureMapA = createEntryBlockAlloca(TheFunction, FutureMapType_, "future_map.alloca");
  store(FutureMapRV, FutureMapA);
  
	//----------------------------------------------------------------------------
  // Destroy argument map
  
  destroyOpaqueType(TheModule, ArgMapA, "legion_argument_map_destroy", "arg_map");
  
	//----------------------------------------------------------------------------
  // Destroy future map
  
  destroyOpaqueType(TheModule, FutureMapA, "legion_future_map_destroy", "future_map");
  

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  destroyOpaqueType(TheModule, LauncherA, "legion_index_launcher_destroy", "task_launcher");
  
  //----------------------------------------------------------------------------
  // Deallocate storate

  destroyGlobalArguments(TheModule, TaskArgsA);

  //return Builder_.CreateLoad(FutureMapType_, FutureMapA);
	return nullptr;
}


//==============================================================================
// get a future value
//==============================================================================
Value* LegionTasker::loadFuture(Module &TheModule, Value* FutureA,
    Type *DataT, Value* DataSizeV)
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

  return Builder_.CreateLoad(DataT, DataA, "future");
}

//==============================================================================
// insert a future value
//==============================================================================
Value* LegionTasker::createFuture(Module &, Function* TheFunction,
    const std::string & Name)
{
  auto FutureA = createEntryBlockAlloca(TheFunction, FutureType_, "future.alloca");
  AbstractTasker::createFuture(Name, FutureA);
  return FutureA;
}

//==============================================================================
// destroey a future value
//==============================================================================
void LegionTasker::destroyFuture(Module &TheModule, Value* FutureA)
{
  destroyOpaqueType(TheModule, FutureA, "legion_future_destroy", "future");
}

//==============================================================================
// Is this a future type
//==============================================================================
bool LegionTasker::isFuture(Value* FutureA) const
{
  auto FutureT = FutureA->getType();
  if (isa<AllocaInst>(FutureA)) FutureT = FutureT->getPointerElementType();
  return (FutureT == FutureType_);
}

}
