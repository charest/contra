#include "config.hpp"

#include "args.hpp"
#include "codegen.hpp"
#include "errors.hpp"
#include "legion.hpp"
#include "legion_rt.hpp"

#include "utils/llvm_utils.hpp"
#include "utils/string_utils.hpp"

#include <legion.h>

#include <vector>

using namespace llvm;
using namespace utils;
  
namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Legion tasker args
////////////////////////////////////////////////////////////////////////////////

cl::opt<std::string> OptionLegion(
    "legion-opts",
    cl::desc("Legion runtime options"),
    cl::cat(OptionCategory));

////////////////////////////////////////////////////////////////////////////////
// Legion tasker
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Constructor
//==============================================================================
LegionTasker::LegionTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  CharType_ = llvmType<char>(getContext());
  OffType_ = llvmType<off_t>(getContext());
  RealmIdType_ = llvmType<realm_id_t>(getContext());
  NumRegionsType_ = llvmType<std::uint32_t>(getContext()); 
  TaskVariantIdType_ = llvmType<legion_variant_id_t>(getContext());
  TaskIdType_ = llvmType<legion_task_id_t>(getContext());
  ProcIdType_ = llvmType<legion_processor_kind_t>(getContext());
  MapperIdType_ = llvmType<legion_mapper_id_t>(getContext()); 
  MappingTagIdType_ = llvmType<legion_mapping_tag_id_t>(getContext());
  FutureIdType_ = llvmType<unsigned>(getContext());
  CoordType_ = llvmType<legion_coord_t>(getContext());
  Point1dType_ = ArrayType::get(CoordType_, 1);
  IndexSpaceIdType_ = llvmType<legion_index_space_id_t>(getContext());
  IndexTreeIdType_ = llvmType<legion_index_tree_id_t>(getContext());
  TypeTagType_ = llvmType<legion_type_tag_t>(getContext());
  FieldSpaceIdType_ = llvmType<legion_field_space_id_t>(getContext());
  FieldIdType_ = llvmType<legion_field_id_t>(getContext());
  RegionTreeIdType_ = llvmType<legion_region_tree_id_t>(getContext());
  IndexPartitionIdType_ = llvmType<legion_index_partition_id_t>(getContext());

  TaskType_ = createOpaqueType("legion_task_t", getContext());
  RegionType_ = createOpaqueType("legion_physical_region_t", getContext());
  ContextType_ = createOpaqueType("legion_context_t", getContext());
  RuntimeType_ = createOpaqueType("legion_runtime_t", getContext());
  ExecSetType_ = createOpaqueType("legion_execution_constraint_set_t", getContext());
  LayoutSetType_ = createOpaqueType("legion_task_layout_constraint_set_t", getContext());
  PredicateType_ = createOpaqueType("legion_predicate_t", getContext());
  TaskLauncherType_ = createOpaqueType("legion_task_launcher_t", getContext());
  IndexLauncherType_ = createOpaqueType("legion_index_launcher_t", getContext());
  FutureType_ = createOpaqueType("legion_future_t", getContext());
  TaskConfigType_ = createTaskConfigOptionsType(getContext());
  TaskArgsType_ = createTaskArgumentsType(getContext());
  DomainPointType_ = createDomainPointType(getContext());
  Rect1dType_ = createRect1dType(getContext());
  DomainRectType_ = createDomainRectType(getContext());
  ArgMapType_ = createOpaqueType("legion_argument_map_t", getContext());
  FutureMapType_ = createOpaqueType("legion_future_map_t", getContext());
  IndexSpaceType_ = createIndexSpaceType(getContext());
  FieldSpaceType_ = createFieldSpaceType(getContext());
  FieldAllocatorType_ = createOpaqueType("legion_field_allocator_t", getContext());
  LogicalRegionType_ = createLogicalRegionType(getContext());
  IndexPartitionType_ = createIndexPartitionType(getContext());
  LogicalPartitionType_ = createLogicalPartitionType(getContext());
  AccessorArrayType_ = createOpaqueType("legion_accessor_array_1d_t", getContext());
  ByteOffsetType_ = createByteOffsetType(getContext());

  IndexSpaceDataType_ = createIndexSpaceDataType(getContext());
  FieldDataType_ = createFieldDataType(getContext());
  AccessorDataType_ = createAccessorDataType(getContext());
  PartitionDataType_ = createPartitionDataType(getContext());
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createOpaqueType(
    const std::string & Name,
    LLVMContext & TheContext)
{
  auto OpaqueType = StructType::create( TheContext, Name );

  std::vector<Type*> members{ VoidPtrType_ }; 
  OpaqueType->setBody( members );

  return OpaqueType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createTaskConfigOptionsType(LLVMContext & TheContext)
{
  std::vector<Type*> members(4, BoolType_);
  auto OptionsType = StructType::create( TheContext, members, "task_config_options_t" );
  return OptionsType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createTaskArgumentsType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { VoidPtrType_, SizeType_ };
  auto NewType = StructType::create( TheContext, members, "legion_task_argument_t" );
  return NewType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createDomainPointType(LLVMContext & TheContext)
{
  auto ArrayT = ArrayType::get(CoordType_, MAX_POINT_DIM); 
  std::vector<Type*> members = { Int32Type_, ArrayT };
  auto NewType = StructType::create( TheContext, members, "legion_domain_point_t" );
  return NewType;
}

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createRect1dType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { Point1dType_, Point1dType_ };
  auto NewType = StructType::create( TheContext, members, "legion_rect_1d_t" );
  return NewType;
}
  

//==============================================================================
// Create the opaque type
//==============================================================================
StructType * LegionTasker::createDomainRectType(LLVMContext & TheContext)
{
  auto ArrayT = ArrayType::get(CoordType_, 2*LEGION_MAX_DIM); 
  std::vector<Type*> members = { RealmIdType_, Int32Type_, ArrayT };
  auto NewType = StructType::create( TheContext, members, "legion_domain_t" );
  return NewType;
}

//==============================================================================
// Create the index space type
//==============================================================================
StructType * LegionTasker::createIndexSpaceType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { IndexSpaceIdType_, IndexTreeIdType_, TypeTagType_ };
  auto NewType = StructType::create( TheContext, members, "legion_index_space_t" );
  return NewType;
}

//==============================================================================
// Create the field space type
//==============================================================================
StructType * LegionTasker::createFieldSpaceType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { FieldSpaceIdType_ };
  auto NewType = StructType::create( TheContext, members, "legion_field_space_t" );
  return NewType;
}

//==============================================================================
// Create the logical region type
//==============================================================================
StructType * LegionTasker::createLogicalRegionType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { RegionTreeIdType_, IndexSpaceType_, FieldSpaceType_ };
  auto NewType = StructType::create( TheContext, members, "legion_logical_region_t" );
  return NewType;
}

//==============================================================================
// Create the logical region type
//==============================================================================
StructType * LegionTasker::createIndexPartitionType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { IndexPartitionIdType_, IndexTreeIdType_, TypeTagType_ };
  auto NewType = StructType::create( TheContext, members, "legion_index_partition_t" );
  return NewType;
}

//==============================================================================
// Create the logical region type
//==============================================================================
StructType * LegionTasker::createLogicalPartitionType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { RegionTreeIdType_, IndexPartitionType_, FieldSpaceType_ };
  auto NewType = StructType::create( TheContext, members, "legion_index_partition_t" );
  return NewType;
}

//==============================================================================
// Create the byte offset type
//==============================================================================
StructType * LegionTasker::createByteOffsetType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { Int32Type_ };
  auto NewType = StructType::create( TheContext, members, "legion_byte_offset_t" );
  return NewType;
}


//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createFieldDataType(LLVMContext & TheContext)
{
  std::vector<Type*> members = {
    IndexSpaceType_,
    FieldSpaceType_,
    FieldAllocatorType_,
    FieldIdType_,
    LogicalRegionType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_field_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createAccessorDataType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { FieldIdType_, RegionType_, LogicalRegionType_, 
    DomainRectType_, Rect1dType_, Rect1dType_, ByteOffsetType_,
    AccessorArrayType_, VoidPtrType_, SizeType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_accessor_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createIndexSpaceDataType(LLVMContext & TheContext)
{
  auto IntT = llvmType<int_t>(getContext());
  std::vector<Type*> members = { IntT, IntT, IntT, IndexSpaceType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_index_space_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * LegionTasker::createPartitionDataType(LLVMContext & TheContext)
{
  std::vector<Type*> members = { IndexPartitionType_, LogicalPartitionType_ };
  auto NewType = StructType::create( TheContext, members, "contra_legion_partition_t" );
  return NewType;
}

  
//==============================================================================
// Create a true predicate
//==============================================================================
AllocaInst* LegionTasker::createPredicateTrue(Module &TheModule)
{
  auto PredicateRT = reduceStruct(PredicateType_, TheModule);
  auto PredicateRV = TheHelper_.callFunction(
      TheModule,
      "legion_predicate_true",
      PredicateRT,
      {},
      "pred_true");
  
  auto PredicateA = TheHelper_.createEntryBlockAlloca(PredicateType_, "predicate.alloca");
  store(PredicateRV, PredicateA);

  return PredicateA;
}

//==============================================================================
// Codegen the global arguments
//==============================================================================
AllocaInst* LegionTasker::createGlobalArguments(
    Module &TheModule,
    const std::vector<Value*> & ArgVorAs)
{
  auto TaskArgsT = TaskArgsType_;
  auto TaskArgsA = TheHelper_.createEntryBlockAlloca(TaskArgsT, "args.alloca");

  //----------------------------------------------------------------------------
  // Identify futures
  
  auto NumArgs = ArgVorAs.size();
  
  std::vector<char> ArgEnums(NumArgs);
  std::vector<unsigned> ValueArgId;
  std::vector<unsigned> FieldArgId;
  
  auto ArgSizesT = ArrayType::get(SizeType_, NumArgs);
  auto ArgSizesA = TheHelper_.createEntryBlockAlloca(ArgSizesT);


  for (unsigned i=0; i<NumArgs; i++) {
    ArgType ArgEnum;
    auto ArgVorA = ArgVorAs[i];
    if (isFuture(ArgVorA)) {
      ArgEnum = ArgType::Future;
    }
    else if (isField(ArgVorA)) {
      ArgEnum = ArgType::Field;
      FieldArgId.emplace_back(i);
    }
    else {
      ValueArgId.emplace_back(i);
      ArgEnum = ArgType::None;
    }
    ArgEnums[i] = static_cast<char>(ArgEnum);
  
    auto ArgSizeV = getSerializedSize(TheModule, ArgVorA, SizeType_);
    TheHelper_.insertValue(ArgSizesA, ArgSizeV, i);
  }

  //----------------------------------------------------------------------------
  // First count sizes

  // add 1 byte for each argument first
  auto ArgSizeT = TaskArgsType_->getElementType(1);
  auto ArgSizeC = llvmValue(getContext(), ArgSizeT, NumArgs);
  TheHelper_.insertValue( TaskArgsA, ArgSizeC, 1);

  // count user argument sizes
  for (auto i : ValueArgId) {
    auto ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 0, 1);
    auto ArgSizeV = TheHelper_.extractValue(ArgSizesT, ArgSizesA, i);
    TheHelper_.increment(ArgSizeT, ArgSizeGEP, ArgSizeV, "addoffset");
  }
  
  // add 8 bytes for each field argument
  auto NumFieldArgs = FieldArgId.size();
  auto ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 0, 1);
  TheHelper_.increment(ArgSizeT, ArgSizeGEP, NumFieldArgs*8, "addoffset");

  //----------------------------------------------------------------------------
  // Allocate storate
 
  auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);
  auto MallocI = TheHelper_.createMalloc(ByteType_, ArgSizeV, "args");
  TheHelper_.insertValue(TaskArgsA, MallocI, 0);
  
  //----------------------------------------------------------------------------
  // create an array with booleans identifying argyment type
  
  auto ArrayGEP = llvmArray(getContext(), TheModule, ArgEnums);
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  TheHelper_.memCopy(ArgDataPtrV, ArrayGEP, llvmValue<size_t>(getContext(), NumArgs)); 
 
  //----------------------------------------------------------------------------
  // Copy args

  // add 1 byte for each argument first
  TheHelper_.insertValue(TaskArgsA, llvmValue(getContext(), ArgSizeT, NumArgs), 1);
  
  for (auto i : ValueArgId) {
    auto ArgV = TheHelper_.getAsAlloca(ArgVorAs[i]);
    // copy
    auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
    serialize(TheModule, ArgV, ArgDataPtrV, ArgSizeT, ArgSizeGEP);
    // increment
    ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 0, 1);
    auto ArgSizeV = TheHelper_.extractValue(ArgSizesT, ArgSizesA, i);
    TheHelper_.increment(ArgSizeT, ArgSizeGEP, ArgSizeV, "addoffset");
  }
  
  //----------------------------------------------------------------------------
  // Add field identifiers
    
  auto FieldDataPtrT = FieldDataType_->getPointerTo();
  auto UnsignedT = llvmType<unsigned>(getContext());
  auto FieldDataF = TheHelper_.createFunction(
      TheModule,
      "contra_legion_pack_field_data",
      VoidType_,
      {FieldDataPtrT, UnsignedT, VoidPtrType_});

  auto ArgDataPtrT = TaskArgsT->getElementType(0);
  unsigned regidx = 0;
  for (auto i : FieldArgId) {
    Value* ArgV = ArgVorAs[i];
    // load offset
    auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);
    // offset data pointer
    auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
    auto OffsetArgDataPtrV = TheHelper_.offsetPointer(ArgDataPtrT, ArgDataPtrV, ArgSizeV);
    // pack field info
    auto ArgSizeGEP = TheHelper_.getElementPointer(TaskArgsA, 0, 1);
    std::vector<Value*> ArgVs = {
      ArgV,
      llvmValue(getContext(), UnsignedT, regidx++),
      OffsetArgDataPtrV};
    getBuilder().CreateCall(FieldDataF, ArgVs);
    // increment
    TheHelper_.increment(ArgSizeT, ArgSizeGEP, 8, "addoffset");
  }
  

  return TaskArgsA; 
}

//==============================================================================
// Codegen the global future arguments
//==============================================================================
void LegionTasker::createGlobalFutures(
    llvm::Module & TheModule,
    AllocaInst* LauncherA,
    const std::vector<Value*> & ArgVorAs,
    bool IsIndex )
{
  auto FutureRT = reduceStruct(FutureType_, TheModule);

  StructType* LauncherT = IsIndex ? IndexLauncherType_ : TaskLauncherType_;
  auto LauncherRT = reduceStruct(LauncherT, TheModule);

  std::vector<Type*> AddFutureArgTs = {LauncherRT, FutureRT};

  std::string FunN = IsIndex ?
    "legion_index_launcher_add_future" : "legion_task_launcher_add_future";
  auto AddFutureF = TheHelper_.createFunction(
      TheModule,
      FunN,
      VoidType_,
      AddFutureArgTs);

  auto NumArgs = ArgVorAs.size();
  for (unsigned i=0; i<NumArgs; i++) {
    auto FutureV = ArgVorAs[i];
    if (!isFuture(FutureV)) continue;
    FutureV = TheHelper_.getAsAlloca(FutureV);
    auto FutureRV = load(FutureType_, FutureV, TheModule, "future");
    auto LauncherRV = load(LauncherA, TheModule, "task_launcher");
    std::vector<Value*> AddFutureArgVs = {LauncherRV, FutureRV};
    getBuilder().CreateCall(AddFutureF, AddFutureArgVs);
  }
}

//==============================================================================
// Create the partition data
//==============================================================================
AllocaInst* LegionTasker::createPartitionInfo(
    llvm::Module & TheModule )
{
  auto PartInfoA = TheHelper_.createEntryBlockAlloca(VoidPtrType_, "indexpartinfo");
  auto NullV = Constant::getNullValue(VoidPtrType_);
  getBuilder().CreateStore(NullV, PartInfoA);
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partitions_create",
      VoidType_,
      {PartInfoA});

  pushPartitionInfo(TheModule, PartInfoA);
    
  return PartInfoA;
}

//==============================================================================
// Create the partition data
//==============================================================================
void LegionTasker::pushPartitionInfo(
    Module & TheModule,
    AllocaInst* PartInfoA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partitions_push",
      VoidType_,
      {PartInfoA});
}

void LegionTasker::popPartitionInfo(
    Module & TheModule,
    AllocaInst* PartInfoA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partitions_pop",
      VoidType_,
      {PartInfoA});
}


//==============================================================================
// Destroy the partition data
//==============================================================================
void LegionTasker::destroyPartitionInfo(
    llvm::Module & TheModule )
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  const auto & PartInfoA = LegionE.PartInfoAlloca;

  if (PartInfoA) {
    std::vector<Value*> FunArgVs = { RuntimeA, ContextA, PartInfoA };
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_partitions_destroy",
        VoidType_,
        FunArgVs);
  }

}


//==============================================================================
// Codegen the field arguments
//==============================================================================
void LegionTasker::createFieldArguments(
    llvm::Module & TheModule,
    Value* LauncherA,
    const std::vector<Value*> & ArgVorAs,
    const std::vector<Value*> & PartVorAs,
    Value* IndexSpaceA,
    Value* PartInfoA )
{
  auto NumArgs = ArgVorAs.size();
  
  //----------------------------------------------------------------------------
  // Add region requirements
  if (IndexSpaceA) {
  
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;

    std::vector<Type*> FunArgTs = {
      RuntimeA->getType(),
      ContextA->getType(),
      IndexLauncherType_->getPointerTo(),
      IndexSpaceA->getType(),
      VoidPtrType_->getPointerTo(),
      IndexPartitionType_->getPointerTo(),
      FieldDataType_->getPointerTo()
    };

    auto FunF = TheHelper_.createFunction(
        TheModule,
        "contra_legion_index_add_region_requirement",
        VoidType_,
        FunArgTs);

    for (unsigned i=0; i<NumArgs; i++) {
      auto FieldA = ArgVorAs[i];
      if (!isField(FieldA)) continue;
      FieldA = TheHelper_.getAsAlloca(FieldA);
      Value* PartA = Constant::getNullValue(IndexPartitionType_->getPointerTo());
      if (PartVorAs[i]) PartA =TheHelper_.getAsAlloca(PartVorAs[i]);
      std::vector<Value*> FunArgVs = {
        RuntimeA,
        ContextA,
        LauncherA,
        IndexSpaceA,
        PartInfoA,
        PartA,
        FieldA
      };
      getBuilder().CreateCall(FunF, FunArgVs);
    }
  }
  //----------------------------------------------------------------------------
  else {
    
    auto LauncherRT = reduceStruct(TaskLauncherType_, TheModule);

    std::vector<Type*> FunArgTs = {
      LauncherRT,
      FieldDataType_->getPointerTo() };

    auto FunF = TheHelper_.createFunction(
        TheModule,
        "contra_legion_task_add_region_requirement",
        VoidType_,
        FunArgTs);
  
    for (unsigned i=0; i<NumArgs; i++) {
      auto FieldV = ArgVorAs[i];
      if (!isField(FieldV)) continue;
      FieldV = TheHelper_.getAsAlloca(FieldV); 
      std::vector<Value*> FunArgVs = {LauncherA, FieldV};
      getBuilder().CreateCall(FunF, FunArgVs);
    }
  }
  //----------------------------------------------------------------------------
  
}


//==============================================================================
// Destroy an opaque type
//==============================================================================
AllocaInst* LegionTasker::createOpaqueType(
    Module& TheModule,
    StructType* OpaqueT,
    const std::string & FuncN,
    const std::string & Name)
{
  auto OpaqueRT = reduceStruct(OpaqueT, TheModule);
  
  Value* OpaqueRV = TheHelper_.callFunction(
      TheModule,
      FuncN,
      OpaqueRT,
      {},
      Name);
  
  auto OpaqueA = TheHelper_.createEntryBlockAlloca(OpaqueT, Name);
  store(OpaqueRV, OpaqueA);
  return OpaqueA;
}
  

//==============================================================================
// Destroy an opaque type
//==============================================================================
void LegionTasker::destroyOpaqueType(
    Module& TheModule,
    Type* OpaqueT,
    Value* OpaqueA,
    const std::string & FuncN,
    const std::string & Name)
{
  auto OpaqueRV = load(OpaqueT, OpaqueA, TheModule, Name);

  TheHelper_.callFunction(
      TheModule,
      FuncN,
      VoidType_,
      {OpaqueRV});
}

void LegionTasker::destroyOpaqueType(
    Module& TheModule,
    AllocaInst* OpaqueA,
    const std::string & FuncN,
    const std::string & Name)
{
  destroyOpaqueType(TheModule, OpaqueA->getAllocatedType(), OpaqueA, FuncN, Name);
}
  
//==============================================================================
// Destroy task arguments
//==============================================================================
void LegionTasker::destroyGlobalArguments(
    AllocaInst* TaskArgsA)
{
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  TheHelper_.createFree(ArgDataPtrV);
}


//==============================================================================
// create registration arguments
//==============================================================================
void LegionTasker::createRegistrationArguments(
    Module& TheModule,
    const TaskInfo & Task,
    llvm::AllocaInst *& ExecSetA,
    llvm::AllocaInst *& LayoutSetA,
    llvm::AllocaInst *& TaskConfigA)
{
  //----------------------------------------------------------------------------
  // execution_constraint_set
  
  ExecSetA = createOpaqueType(TheModule, ExecSetType_,
      "legion_execution_constraint_set_create", "execution_constraint");
  
  //----------------------------------------------------------------------------
  // add constraint
  
  auto ProcIdV = llvmValue(getContext(), ProcIdType_, LOC_PROC);  
  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraint");
  std::vector<Value*> AddExecArgVs = {ExecSetV, ProcIdV};
  
  TheHelper_.callFunction(
      TheModule,
      "legion_execution_constraint_set_add_processor_constraint",
      VoidType_,
      AddExecArgVs);

  //----------------------------------------------------------------------------
  // task_layout_constraint_set
 
  LayoutSetA = createOpaqueType(TheModule, LayoutSetType_,
      "legion_task_layout_constraint_set_create", "layout_constraint"); 
  
  //----------------------------------------------------------------------------
  // options
  TaskConfigA = TheHelper_.createEntryBlockAlloca(TaskConfigType_, "options");
  auto BoolT = TaskConfigType_->getElementType(0);
  auto FalseV = Constant::getNullValue(BoolT);
  TheHelper_.memSet(TaskConfigA, FalseV, 4); 

  if (Task.isLeaf()) {
    auto TrueV = llvmValue(getContext(), BoolT, 1);
    TheHelper_.insertValue(TaskConfigA, TrueV, 0);
  }

}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    bool IsIndex)
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
  BasicBlock *BB = BasicBlock::Create(getContext(), "entry", WrapperF);
  getBuilder().SetInsertPoint(BB);

  // create the context
  auto & LegionE = startTask();
  auto & ContextA = LegionE.ContextAlloca;
  auto & RuntimeA = LegionE.RuntimeAlloca;
  ContextA = TheHelper_.createEntryBlockAlloca(WrapperF, ContextType_, "context.alloca");
  RuntimeA = TheHelper_.createEntryBlockAlloca(WrapperF, RuntimeType_, "runtime.alloca");
  
  auto RealT = llvmType<real_t>(getContext());
  LegionE.TimerAlloca = TheHelper_.createEntryBlockAlloca(RealT);
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_timer_start",
      VoidType_,
      {LegionE.TimerAlloca});


  // allocate arguments
  std::vector<Value*> WrapperArgVs;
  WrapperArgVs.reserve(WrapperArgTs.size());

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgT = WrapperArgTs[ArgIdx];
    // Create an alloca for this variable.
    auto ArgN = std::string(Arg.getName()) + ".alloca";
    auto Alloca = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgN);
    // Store the initial value into the alloca.
    getBuilder().CreateStore(&Arg, Alloca);
    WrapperArgVs.emplace_back(Alloca);
    ArgIdx++;
  }

  // loads
  auto DataV = TheHelper_.load(WrapperArgVs[0], "data");
  auto DataLenV = TheHelper_.load(WrapperArgVs[1], "datalen");
  //auto UserDataV = getBuilder().CreateLoad(VoidPtrType_, WrapperArgVs[2], "userdata");
  //auto UserLenV = getBuilder().CreateLoad(SizeType_, WrapperArgVs[3], "userlen");
  auto ProcIdV = TheHelper_.load(WrapperArgVs[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto TaskA = TheHelper_.createEntryBlockAlloca(WrapperF, TaskType_, "task.alloca");
 
  auto RegionsT = RegionType_->getPointerTo();
  auto RegionsA = TheHelper_.createEntryBlockAlloca(WrapperF, RegionsT, "regions.alloca");
  auto NullV = Constant::getNullValue(RegionsT);
  getBuilder().CreateStore(NullV, RegionsA);

  auto NumRegionsA = TheHelper_.createEntryBlockAlloca(WrapperF, NumRegionsType_, "num_regions");
  auto ZeroV = llvmValue(getContext(), NumRegionsType_, 0);
  getBuilder().CreateStore(ZeroV, NumRegionsA);
 
  // args
  std::vector<Value*> PreambleArgVs = {
    DataV,
    DataLenV,
    ProcIdV,
    TaskA,
    RegionsA,
    NumRegionsA,
    ContextA,
    RuntimeA };
  TheHelper_.callFunction(
      TheModule,
      "legion_task_preamble",
      VoidType_,
      PreambleArgVs);
  
  //----------------------------------------------------------------------------
  // Get task args

  auto TaskRV = load(TaskA, TheModule, "task");
  Value* TaskArgsV = TheHelper_.callFunction(
      TheModule,
      "legion_task_get_args",
      VoidPtrType_,
      {TaskRV},
      "args");
  
  auto TaskArgsT = VoidPtrType_;
  auto TaskArgsA = TheHelper_.createEntryBlockAlloca(WrapperF, TaskArgsT, "args.alloca");
  getBuilder().CreateStore(TaskArgsV, TaskArgsA);
  
  //----------------------------------------------------------------------------
  // Allocas for task arguments

  auto NumArgs = TaskArgTs.size();
  std::vector<AllocaInst*> TaskArgAs;
  for (unsigned i=0; i<NumArgs; ++i) {
    auto ArgN = TaskArgNs[i];
    auto ArgStr = ArgN + ".alloca";
    auto ArgT = TaskArgTs[i];
    if (isRange(ArgT) && IsIndex) ArgT = IndexPartitionType_;
    auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, ArgT, ArgStr);
    TaskArgAs.emplace_back(ArgA);
  }
  
  //----------------------------------------------------------------------------
  // get user types

  auto ArrayT = ArrayType::get(CharType_, NumArgs);
  auto ArrayA = TheHelper_.createEntryBlockAlloca(WrapperF, ArrayT, "isfuture");
  auto ArgDataPtrV = TheHelper_.load(TaskArgsA, "args");
  TheHelper_.memCopy(ArrayA, ArgDataPtrV, llvmValue<size_t>(getContext(), NumArgs)); 
  
  //----------------------------------------------------------------------------
  // unpack user variables
  
  auto OffsetT = SizeType_;
  auto OffsetA = TheHelper_.createEntryBlockAlloca(WrapperF, OffsetT, "offset.alloca");
  getBuilder().CreateStore( llvmValue(getContext(), OffsetT, NumArgs), OffsetA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(getContext(), "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(getContext(), "ifcont");
    // evaluate the condition
    auto ArgTypeV = TheHelper_.extractValue(ArrayT, ArrayA, i);
    auto EnumV = llvmValue(getContext(), ArgTypeV->getType(), static_cast<char>(ArgType::None));
    auto CondV = getBuilder().CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    getBuilder().CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    getBuilder().SetInsertPoint(ThenBB);
    // copy
    auto ArgSizeV = deserialize(TheModule, TaskArgAs[i], TaskArgsA, TaskArgsT, OffsetA);
    // increment
    TheHelper_.increment(OffsetA, ArgSizeV, "offset");
    // finish then
    ThenBB->getFirstNonPHI();
    getBuilder().CreateBr(MergeBB);
    ThenBB = getBuilder().GetInsertBlock();
    // Emit merge block.
    WrapperF->insert(WrapperF->end(), MergeBB);
    getBuilder().SetInsertPoint(MergeBB);
  }

  //----------------------------------------------------------------------------
  // partition any ranges
  if (IsIndex) { 
    
    std::vector<Type*> GetRangeArgTs = {
      RuntimeType_->getPointerTo(),
      ContextType_->getPointerTo(),
      TaskType_->getPointerTo(),
      IndexPartitionType_->getPointerTo(),
      IndexSpaceDataType_->getPointerTo()
    };

    auto GetRangeF = TheHelper_.createFunction(
        TheModule,
        "contra_legion_index_space_create_from_index_partition",
        VoidType_,
        GetRangeArgTs);

    for (unsigned i=0; i<NumArgs; i++) {
      auto ArgN = TaskArgNs[i];
      if (isRange(TaskArgTs[i])) {
        auto ArgA = TheHelper_.createEntryBlockAlloca(WrapperF, IndexSpaceDataType_, ArgN);
        getBuilder().CreateCall(GetRangeF, {RuntimeA, ContextA, TaskA, TaskArgAs[i], ArgA});
        TaskArgAs[i] = ArgA;
      }
    }
  
  }
  

  //----------------------------------------------------------------------------
  // unpack future variables
  
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto TaskRT = reduceStruct(TaskType_, TheModule);
  std::vector<Type*> GetFutureArgTs = {TaskRT, FutureIdType_};
  auto GetFutureF = TheHelper_.createFunction(
      TheModule,
      "legion_task_get_future",
      FutureRT,
      GetFutureArgTs);
  
  auto FutureIndexT = FutureIdType_;
  auto FutureIndexA = TheHelper_.createEntryBlockAlloca(WrapperF, FutureIndexT, "futureid.alloca");
  getBuilder().CreateStore( Constant::getNullValue(FutureIndexT), FutureIndexA );
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(getContext(), "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(getContext(), "ifcont");
    // evaluate the condition
    auto ArgTypeV = TheHelper_.extractValue(ArrayT, ArrayA, i);
    auto EnumV = llvmValue(getContext(), ArgTypeV->getType(), static_cast<char>(ArgType::Future));
    auto CondV = getBuilder().CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    getBuilder().CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    getBuilder().SetInsertPoint(ThenBB);
    // get future
    auto TaskRV = load(TaskA, TheModule, "task");
    auto FutureIndexV = TheHelper_.load(FutureIndexA, "futureid");
    std::vector<Value*> GetFutureArgVs = {TaskRV, FutureIndexV};
    auto FutureRV = getBuilder().CreateCall(GetFutureF, GetFutureArgVs, "get_future");
    auto FutureA = TheHelper_.createEntryBlockAlloca(WrapperF, FutureType_, "future");
    store(FutureRV, FutureA);
    // unpack
    auto ArgA = TaskArgAs[i];
    auto ArgT = ArgA->getAllocatedType(); //->getPointerElementType();
    // copy
    auto ArgV = loadFuture(TheModule, FutureA, ArgT);
    getBuilder().CreateStore( ArgV, ArgA );
    // consume the future
    destroyFuture(TheModule, FutureA);
    // increment
    TheHelper_.increment(FutureIndexA, 1, "futureid");
    // finish then
    ThenBB->getFirstNonPHI();
    getBuilder().CreateBr(MergeBB);
    ThenBB = getBuilder().GetInsertBlock();
    // Emit merge block.
    WrapperF->insert(WrapperF->end(), MergeBB);
    getBuilder().SetInsertPoint(MergeBB);
  }
  
  //----------------------------------------------------------------------------
  // unpack Field variables
      
  auto UInt32Type = llvmType<uint32_t>(getContext());
  auto UInt32PtrType = UInt32Type->getPointerTo();
  auto GetFieldDataF = TheHelper_.createFunction(
      TheModule,
      "contra_legion_unpack_field_data",
      VoidType_,
      {VoidPtrType_, UInt32PtrType, UInt32PtrType});
    
  auto AccessorDataPtrT = AccessorDataType_->getPointerTo();
  std::vector<Type*> GetFieldArgTs = {
    RuntimeA->getType(),
    ContextA->getType(),
    RegionsA->getType(),
    NumRegionsA->getType(),
    UInt32PtrType,
    UInt32PtrType,
    AccessorDataPtrT};
  auto GetFieldF = TheHelper_.createFunction(
      TheModule,
      "contra_legion_get_accessor",
      VoidType_,
      GetFieldArgTs);

  auto FieldIndexT = FieldIdType_;
  auto FieldIndexA = TheHelper_.createEntryBlockAlloca(WrapperF, FieldIndexT, "field.alloca");
  getBuilder().CreateStore( Constant::getNullValue(FieldIndexT), FieldIndexA );
  
  auto FieldIdA = TheHelper_.createEntryBlockAlloca(WrapperF, UInt32Type, "fieldid.alloca");
  auto RegionIdA = TheHelper_.createEntryBlockAlloca(WrapperF, UInt32Type, "regid.alloca");
  
  for (unsigned i=0; i<NumArgs; i++) {
    // Create blocks for the then and else cases.
    BasicBlock * ThenBB = BasicBlock::Create(getContext(), "then", WrapperF);
    BasicBlock * MergeBB = BasicBlock::Create(getContext(), "ifcont");
    // evaluate the condition
    auto ArgTypeV = TheHelper_.extractValue(ArrayT, ArrayA, i);
    auto EnumV = llvmValue(getContext(), ArgTypeV->getType(), static_cast<char>(ArgType::Field));
    auto CondV = getBuilder().CreateICmpEQ(ArgTypeV, EnumV, "argtype");
    getBuilder().CreateCondBr(CondV, ThenBB, MergeBB);
    // Emit then block
    getBuilder().SetInsertPoint(ThenBB);
    // unpack the field data
    auto ArgGEP = TheHelper_.offsetPointer(TaskArgsT, TaskArgsA, OffsetA);
    getBuilder().CreateCall( GetFieldDataF, std::vector<Value*>{ArgGEP, FieldIdA, RegionIdA} );
    // get field pointer
    auto ArgPtrV = TheHelper_.createBitCast(TaskArgAs[i], AccessorDataPtrT);
    std::vector<Value*> GetFieldArgVs = {
      RuntimeA,
      ContextA,
      RegionsA,
      NumRegionsA,
      RegionIdA,
      FieldIdA,
      ArgPtrV };
    getBuilder().CreateCall( GetFieldF, GetFieldArgVs );
    // increment
    TheHelper_.increment(FieldIndexA, 1, "fieldid");
    TheHelper_.increment(OffsetA, 8, "offset");
    // finish then
    ThenBB->getFirstNonPHI();
    getBuilder().CreateBr(MergeBB);
    ThenBB = getBuilder().GetInsertBlock();
    // Emit merge block.
    WrapperF->insert(WrapperF->end(), MergeBB);
    getBuilder().SetInsertPoint(MergeBB);
  }

  //----------------------------------------------------------------------------
  // If this is an index task
  
  AllocaInst* IndexA = nullptr;
  if (IsIndex) {
     
    auto TaskRV = load(TaskA, TheModule, "task");
    auto DomainPointA = TheHelper_.createEntryBlockAlloca(WrapperF, DomainPointType_, "domain_point.alloca");

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
 
    getBuilder().CreateCall(GetIndexF, GetIndexArgVs);

    auto PointDataT = DomainPointType_->getElementType(1);
    auto PointDataGEP = TheHelper_.getElementPointer(DomainPointA, 0, 1);
    auto PointDataV = getBuilder().CreateLoad(PointDataT, PointDataGEP);
    auto IndexV = getBuilder().CreateExtractValue(PointDataV, 0);

    IndexA = TheHelper_.createEntryBlockAlloca(WrapperF, llvmType<int_t>(getContext()), "index");
    getBuilder().CreateStore( IndexV, IndexA );
  }

  //----------------------------------------------------------------------------
  // Function body
  return {WrapperF, TaskArgAs, IndexA}; 

}


//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs,
    llvm::Type*)
{
  return taskPreamble(TheModule, TaskName, TaskArgNs, TaskArgTs, true);
}

//==============================================================================
// Create the function wrapper
//==============================================================================
LegionTasker::PreambleResult LegionTasker::taskPreamble(
    Module &TheModule,
    const std::string & Name,
    Function* TaskF)
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

  return taskPreamble(TheModule, TaskName, TaskArgNs, TaskArgTs, false);
}
  
//==============================================================================
// Create the function wrapper
//==============================================================================
void LegionTasker::taskPostamble(Module &TheModule, Value* ResultV, bool)
{

  Value* RetvalV = Constant::getNullValue(VoidPtrType_);
  Value* RetsizeV = llvmValue<std::size_t>(getContext(), 0);

  AllocaInst* RetvalA = nullptr;
  auto RetvalT = VoidPtrType_;

  
  //----------------------------------------------------------------------------
  // Have return value
  bool HasNonVoidResult = ResultV && !ResultV->getType()->isVoidTy();
  if (HasNonVoidResult) {

    // store/load result
    auto ResultA = TheHelper_.getAsAlloca( ResultV );
    ResultV = TheHelper_.getAsValue( ResultV );

    // return size
    auto RetsizeT = RetsizeV->getType();
    RetsizeV = getSerializedSize(TheModule, ResultV, RetsizeT);
    auto RetsizeA = TheHelper_.createEntryBlockAlloca(RetsizeT, "retsize");
    getBuilder().CreateStore( RetsizeV, RetsizeA );

    // allocate space for return value
    RetsizeV = TheHelper_.load(RetsizeA);
    auto MallocI = TheHelper_.createMalloc(ByteType_, RetsizeV, "retval");
    RetvalA = TheHelper_.createEntryBlockAlloca(RetvalT, "retval");
    getBuilder().CreateStore(MallocI, RetvalA );

    // copy data
    RetvalV = TheHelper_.load(RetvalA);
    RetsizeV = TheHelper_.load(RetsizeA);
    TheHelper_.memCopy(RetvalV, ResultA, RetsizeV); 
    serialize(TheModule, ResultA, RetvalV, RetvalT);

    // final loads
    RetsizeV = TheHelper_.load(RetsizeA);
    RetvalV = TheHelper_.load(RetvalA);
    
  }
  
  //----------------------------------------------------------------------------
  // Destroy Index partitions

  destroyPartitionInfo(TheModule);

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
  TheHelper_.callFunction(
      TheModule,
      "legion_task_postamble",
      VoidType_,
      PostambleArgVs);
  
  //----------------------------------------------------------------------------
  // Free memory
  if (HasNonVoidResult) {
    RetvalV = TheHelper_.load(RetvalA);
    TheHelper_.createFree(RetvalV);
  }

  TheHelper_.callFunction(
      TheModule,
      "contra_legion_timer_stop",
      VoidType_,
      {LegionE.TimerAlloca});
  
  // Finish off the function.  Tasks always return void
  getBuilder().CreateRetVoid();

  finishTask();
}

//==============================================================================
// Postregister tasks
//==============================================================================
void LegionTasker::postregisterTask(
    Module &TheModule,
    const std::string & Name,
    const TaskInfo & Task )
{

  //----------------------------------------------------------------------------
  // arguments
  
  llvm::AllocaInst *ExecSetA, *LayoutSetA, *TaskConfigA;
  createRegistrationArguments(
      TheModule,
      Task,
      ExecSetA,
      LayoutSetA,
      TaskConfigA);
  
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunction()->getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(getContext(), TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(getContext(), TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(getContext(), TheModule, Name + " task");
  auto VariantNameV = llvmString(getContext(), TheModule, Name + " variant");

  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(getContext(), 0);
  
  auto BoolT = TaskConfigType_->getElementType(0);
  auto TrueV = Constant::getNullValue(BoolT);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, TrueV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreRetT = TaskVariantIdType_;
  TaskIdV = TheHelper_.callFunction(
      TheModule,
      "legion_runtime_register_task_variant_fnptr",
      PreRetT,
      PreArgVs,
      "task_variant_id");
  
  //----------------------------------------------------------------------------
  // Register reduction
  if (Task.hasReduction()) registerReductionOp(TheModule, Task.getReduction());
}

//==============================================================================
// Preregister tasks
//==============================================================================
void LegionTasker::preregisterTask(
    Module &TheModule,
    const std::string & Name,
    const TaskInfo & Task )
{

  //----------------------------------------------------------------------------
  // arguments
  
  llvm::AllocaInst *ExecSetA, *LayoutSetA, *TaskConfigA;
  createRegistrationArguments(
      TheModule,
      Task,
      ExecSetA,
      LayoutSetA,
      TaskConfigA);
  
  //----------------------------------------------------------------------------
  // registration
  
  auto TaskT = Task.getFunctionType();
  auto TaskF = TheModule.getOrInsertFunction(Task.getName(), TaskT).getCallee();
  
  Value* TaskIdV = llvmValue(getContext(), TaskIdType_, Task.getId());
  auto TaskIdVariantV = llvmValue(getContext(), TaskVariantIdType_, AUTO_GENERATE_ID);
  auto TaskNameV = llvmString(getContext(), TheModule, Name + " task");
  auto VariantNameV = llvmString(getContext(), TheModule, Name + " variant");

  auto ExecSetV = load(ExecSetA, TheModule, "execution_constraints");
  auto LayoutSetV = load(LayoutSetA, TheModule, "layout_constraints");
 
  auto TaskConfigV = load(TaskConfigA, TheModule, "options");

  auto UserDataV = Constant::getNullValue(VoidPtrType_);
  auto UserLenV = llvmValue<std::size_t>(getContext(), 0);

  std::vector<Value*> PreArgVs = { TaskIdV, TaskIdVariantV, TaskNameV,
    VariantNameV, ExecSetV, LayoutSetV, TaskConfigV, TaskF, UserDataV, UserLenV };
  
  auto PreRetT = TaskVariantIdType_;
  TaskIdV = TheHelper_.callFunction(
      TheModule,
      "legion_runtime_preregister_task_variant_fnptr",
      PreRetT,
      PreArgVs,
      "task_variant_id");
  
  destroyOpaqueType(TheModule, ExecSetA, "legion_execution_constraint_set_destroy",
      "exec_set");
  destroyOpaqueType(TheModule, LayoutSetA, "legion_task_layout_constraint_set_destroy",
      "layout_set");
  
  //----------------------------------------------------------------------------
  // Register reduction
  if (Task.hasReduction()) registerReductionOp(TheModule, Task.getReduction());
}
  
//==============================================================================
// Set top level task
//==============================================================================
void LegionTasker::setTopLevelTask(Module &TheModule, const TaskInfo & TaskI)
{

  auto TaskId = TaskI.getId();
  auto TaskIdV = llvmValue(getContext(), TaskIdType_, TaskId);
  std::vector<Value*> SetArgVs = { TaskIdV };
  TheHelper_.callFunction(
      TheModule,
      "legion_runtime_set_top_level_task_id",
      VoidType_,
      SetArgVs);
}
  
//==============================================================================
// start runtime
//==============================================================================
void LegionTasker::startRuntime(Module &TheModule)
{
  // setup backend args
  std::vector<std::string> Argv  = {"./contra"};

  auto SplitArgs = utils::split(OptionLegion, ' ');
  for (const auto & A : SplitArgs) 
    Argv.emplace_back(A);
  
  bool HasGsize = false;
  for (const auto & A : Argv) {
    if (A == "-ll:gsize") {
      HasGsize = true;
      break;
    }
  }

  if (!HasGsize) {
    Argv.emplace_back("-ll:gsize");
    Argv.emplace_back("0");
  }
  
  // startup runtime
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_startup",
      VoidType_);

  auto ArgcV = llvmValue(getContext(), Int32Type_, Argv.size());

  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argv.size(); ++i)
    ArgVs.emplace_back( llvmString(getContext(), TheModule, Argv[i]) );

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(getContext()));
  auto ArgvV = llvmArray(getContext(), TheModule, ArgVs, {ZeroC, ZeroC});
  
  auto BackV = llvmValue(getContext(), BoolType_, false);

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV, BackV };
  TheHelper_.callFunction(
      TheModule,
      "legion_runtime_start",
      Int32Type_,
      StartArgVs,
      "start");
  
}

//==============================================================================
// Launch a task
//==============================================================================
Value* LegionTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    const std::vector<Value*> & ArgVs)
{
  //----------------------------------------------------------------------------
  // Global arguments
  std::vector<unsigned> FutureArgId;
  std::vector<unsigned> FieldArgId;
  auto TaskArgsA = createGlobalArguments(TheModule, ArgVs);
  
 
  //----------------------------------------------------------------------------
  // Predicate
  auto PredicateA = createPredicateTrue(TheModule);
  
  //----------------------------------------------------------------------------
  // Launch

  auto TaskId = TaskI.getId();
  auto MapperIdV = llvmValue(getContext(), MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(getContext(), MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(getContext(), TaskIdType_, TaskId);
  
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);

  auto LauncherRT = reduceStruct(TaskLauncherType_, TheModule);

  std::vector<Value*> LaunchArgVs = {TaskIdV, ArgDataPtrV, ArgSizeV, 
    PredicateV, MapperIdV, MappingTagIdV};
  Value* LauncherRV = TheHelper_.callFunction(
      TheModule,
      "legion_task_launcher_create",
      LauncherRT,
      LaunchArgVs,
      "launcher_create");
  auto LauncherA = TheHelper_.createEntryBlockAlloca(TaskLauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Add futures
  createGlobalFutures(TheModule, LauncherA, ArgVs, false );
  
  //----------------------------------------------------------------------------
  // Add fields
  createFieldArguments(TheModule, LauncherA, ArgVs );

  
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
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto FutureRV = TheHelper_.callFunction(
      TheModule,
      "legion_task_launcher_execute",
      FutureRT,
      ExecArgVs,
      "launcher_exec");
  auto FutureA = TheHelper_.createEntryBlockAlloca(FutureType_, "future.alloca");
  store(FutureRV, FutureA);

  //----------------------------------------------------------------------------
  // Destroy launcher
  
  destroyOpaqueType(TheModule, LauncherA, "legion_task_launcher_destroy", "task_launcher");
  
  //----------------------------------------------------------------------------
  // Deallocate storate
  destroyGlobalArguments(TaskArgsA);

  return FutureA;
}

//==============================================================================
// Launch an index task
//==============================================================================
Value* LegionTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    std::vector<Value*> ArgAs,
    const std::vector<Value*> & PartAs,
    Value* RangeV,
    const AbstractReduceInfo* AbstractRedop)
{
  auto RealT = llvmType<real_t>(getContext());
  auto TimerA = TheHelper_.createEntryBlockAlloca(RealT);
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_timer_start",
      VoidType_,
      {TimerA});
  
  // temporaries
  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);
  pushPartitionInfo(TheModule, PartInfoA);
  
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
        RuntimeA,
        ContextA,
        IndexSpaceA,
        IndexPartitionA,
        PartInfoA};
      TheHelper_.callFunction(
          TheModule,
          "contra_legion_register_index_partition",
          VoidType_,
          FunArgVs);
    }
  }


  //----------------------------------------------------------------------------
  // Global arguments
  std::vector<unsigned> FutureArgId;
  std::vector<unsigned> FieldArgId;
  auto TaskArgsA = createGlobalArguments(TheModule, ArgAs);

  //----------------------------------------------------------------------------
  // Predicate
  auto PredicateA = createPredicateTrue(TheModule);
  
  //----------------------------------------------------------------------------
  // Create domain  

  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  auto DomainRectA = TheHelper_.createEntryBlockAlloca(DomainRectType_, "domain");

  std::vector<Value*> DomainFromArgVs = { RuntimeA, IndexSpaceA, DomainRectA };
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_domain_create",
      VoidType_,
      DomainFromArgVs);
  
  //----------------------------------------------------------------------------
  // argument map 
  
  auto ArgMapA = createOpaqueType(TheModule, ArgMapType_,
      "legion_argument_map_create", "arg_map");
  
  //----------------------------------------------------------------------------
  // Launch
 
  auto TaskId = TaskI.getId();
  auto MapperIdV = llvmValue(getContext(), MapperIdType_, 0); 
  auto MappingTagIdV = llvmValue(getContext(), MappingTagIdType_, 0); 
  auto PredicateV = load(PredicateA, TheModule, "predicate");
  auto TaskIdV = llvmValue(getContext(), TaskIdType_, TaskId);
  auto MustV = Constant::getNullValue(BoolType_);
  auto ArgMapV = load(ArgMapA, TheModule, "arg_map");
  
  auto ArgDataPtrV = TheHelper_.extractValue(TaskArgsA, 0);
  auto ArgSizeV = TheHelper_.extractValue(TaskArgsA, 1);

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

  Value* LauncherRV = getBuilder().CreateCall(LaunchF, LaunchArgVs, "launcher_create");
  auto LauncherA = TheHelper_.createEntryBlockAlloca(IndexLauncherType_, "task_launcher.alloca");
  store(LauncherRV, LauncherA);
  
  //----------------------------------------------------------------------------
  // Add futures
  createGlobalFutures(TheModule, LauncherA, ArgAs, true );
 
  //----------------------------------------------------------------------------
  // Add fields
  createFieldArguments(
      TheModule,
      LauncherA,
      ArgAs,
      PartAs,
      IndexSpaceA,
      PartInfoA);
  
  //TheHelper_.callFunction(
  //    TheModule,
  //    "contra_legion_timer_stop",
  //    VoidType_,
  //    {TimerA});
  
  //----------------------------------------------------------------------------
  // Execute
  
  LauncherRV = load(LauncherA, TheModule, "task_launcher");
  auto ContextV = load(ContextA, TheModule, "context");
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  std::vector<Value*> ExecArgVs = { RuntimeV, ContextV, LauncherRV };

  Value* Result = nullptr;

  //------------------------------------
  // With reduction
  if (AbstractRedop) {
    auto Redop = dynamic_cast<const LegionReduceInfo*>(AbstractRedop);
    auto RedopId = Redop->getId();
    auto OpIdC = llvmValue<legion_reduction_op_id_t>(getContext(), RedopId);
    ExecArgVs.emplace_back( OpIdC );
    auto FutureRT = reduceStruct(FutureType_, TheModule);
    Value* FutureRV = TheHelper_.callFunction(
        TheModule,
        "legion_index_launcher_execute_reduction",
        FutureRT,
        ExecArgVs,
        "launcher_exec");
    auto FutureA = TheHelper_.createEntryBlockAlloca(FutureType_, "future.alloca");
    store(FutureRV, FutureA);
    Result = FutureA;
  }
  //------------------------------------
  // No reduction
  else {
    auto FutureMapRT = reduceStruct(FutureMapType_, TheModule);
    Value* FutureMapRV = TheHelper_.callFunction(
        TheModule,
        "legion_index_launcher_execute",
        FutureMapRT,
        ExecArgVs,
        "launcher_exec");
    auto FutureMapA = TheHelper_.createEntryBlockAlloca(FutureMapType_, "future_map.alloca");
    store(FutureMapRV, FutureMapA);
  
    // Destroy future map
    //TheHelper_.callFunction(
    //    TheModule,
    //    "legion_future_map_wait_all_results",
    //    VoidType_,
    //    {FutureMapRV});
    destroyOpaqueType(TheModule, FutureMapA, "legion_future_map_destroy", "future_map");
  }
	
  //----------------------------------------------------------------------------
  // Destroy argument map
  
  destroyOpaqueType(TheModule, ArgMapA, "legion_argument_map_destroy", "arg_map");

  
  //----------------------------------------------------------------------------
  // cleanup
  
  destroyPartitions(TheModule, TempParts);
  
  destroyOpaqueType(TheModule, LauncherA, "legion_index_launcher_destroy", "task_launcher");
  
  popPartitionInfo(TheModule, PartInfoA);
  
  destroyGlobalArguments(TaskArgsA);

  return Result;
}


//==============================================================================
// get a future value
//==============================================================================
Value* LegionTasker::loadFuture(
    Module &TheModule,
    Value* Future,
    Type *DataT)
{
  // args
  auto FutureA = TheHelper_.getAsAlloca(Future);
  auto FutureRV = load(FutureA, TheModule, "future");
  Value* DataPtrV = TheHelper_.callFunction(
      TheModule,
      "legion_future_get_untyped_pointer",
      VoidPtrType_,
      {FutureRV},
      "future");
  
  auto DataA = TheHelper_.createEntryBlockAlloca(DataT);
  deserialize(TheModule, DataA, DataPtrV, VoidPtrType_);

  return TheHelper_.load(DataA, "future");
}

//==============================================================================
// destroey a future value
//==============================================================================
void LegionTasker::destroyFuture(Module &TheModule, Value* FutureA)
{
  destroyOpaqueType(TheModule, FutureType_, FutureA, "legion_future_destroy", "future");
}
  
//==============================================================================
// copy a value into a future
//==============================================================================
void LegionTasker::toFuture(
    Module & TheModule,
    Value* ValueV,
    Value* FutureA)
{
  // load runtime
  const auto & LegionE = getCurrentTask();
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto RuntimeV = load(RuntimeA, TheModule, "runtime");

  auto ValueT = ValueV->getType();
  auto ValueA = TheHelper_.createEntryBlockAlloca(ValueT);
  getBuilder().CreateStore( ValueV, ValueA );

  auto ValuePtrV = TheHelper_.createBitCast(ValueA, VoidPtrType_);

  auto ValueSizeV = TheHelper_.getTypeSize<size_t>(ValueT);

  std::vector<Value*> FunArgVs = {RuntimeV, ValuePtrV, ValueSizeV};
    
  auto FutureRT = reduceStruct(FutureType_, TheModule);
  auto FutureRV = TheHelper_.callFunction(
      TheModule,
      "legion_future_from_untyped_pointer",
      FutureRT,
      FunArgVs,
      "future");
  store(FutureType_, FutureRV, FutureA);

}

//==============================================================================
// copy a future
//==============================================================================
void LegionTasker::copyFuture(
    Module & TheModule,
    Value* Val,
    Value* FutureA)
{
  // load runtime
  auto FutureRT = reduceStruct(FutureType_, TheModule);

  auto ValueRV = TheHelper_.extractValue(Val, 0);
  auto FutureRV = TheHelper_.callFunction(
      TheModule,
      "legion_future_copy",
      FutureRT,
      {ValueRV},
      "future");
  store(FutureType_, FutureRV, FutureA);
}

//==============================================================================
// Is this a future type
//==============================================================================
bool LegionTasker::isFuture(Value* FutureV) const
{
  auto FutureT = FutureV->getType();
  if (auto FutureA = dyn_cast<AllocaInst>(FutureV))
    FutureT = FutureA->getAllocatedType();
  return (FutureT == FutureType_);
}

//==============================================================================
// Is this a field type
//==============================================================================
bool LegionTasker::isField(Value* FieldV) const
{
  auto FieldT = FieldV->getType();
  if (auto FieldA = dyn_cast<AllocaInst>(FieldV))
    FieldT = FieldA->getAllocatedType();
  return (FieldT == FieldDataType_);
}


//==============================================================================
// Create a legion field
//==============================================================================
void LegionTasker::createField(
    Module & TheModule,
    Value* FieldA,
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  
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
    RuntimeA,
    ContextA,
    NameV,
    DataSizeV, 
    VarV,
    IndexSpaceA,
    FieldA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void LegionTasker::destroyField(Module &TheModule, Value* FieldA)
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  std::vector<Value*> FunArgVs = {RuntimeA, ContextA, FieldA};
    
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_field_destroy",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Is this an range type
//==============================================================================
bool LegionTasker::isRange(Type* RangeT) const
{
  return (RangeT == IndexSpaceDataType_);
}

bool LegionTasker::isRange(Value* RangeV) const
{
  auto RangeT = RangeV->getType();
  if (auto RangeA = dyn_cast<AllocaInst>(RangeV))
    RangeT = RangeA->getAllocatedType();
  return isRange(RangeT);
}


//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createRange(
    Module & TheModule,
    const std::string & Name,
    Value* StartV,
    Value* EndV,
    Value* StepV)
{
  auto IndexSpaceA = TheHelper_.createEntryBlockAlloca(IndexSpaceDataType_, "index");

  StartV = TheHelper_.getAsValue(StartV);
  EndV = TheHelper_.getAsValue(EndV);
  if (StepV) StepV = TheHelper_.getAsValue(StepV);

  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;
    
    auto NameV = llvmString(getContext(), TheModule, Name);

    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, NameV, StartV, EndV, IndexSpaceA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_create",
        VoidType_,
        FunArgVs);
  }
  else { 
    TheHelper_.insertValue(IndexSpaceA, StartV, 0);
    auto OneC = llvmValue<int_t>(getContext(), 1);
    EndV = getBuilder().CreateAdd(EndV, OneC);
    TheHelper_.insertValue(IndexSpaceA, EndV, 1);
    if (!StepV) StepV = OneC;
    TheHelper_.insertValue(IndexSpaceA, StepV, 2);
  }

  
  return IndexSpaceA;

}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createPartition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* IndexPartitionA,
    Value* ValueA)
{
  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;
  auto & PartInfoA = LegionE.PartInfoAlloca;

  if (!PartInfoA) PartInfoA = createPartitionInfo(TheModule);
  
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);
  
  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);

  //------------------------------------
  if (isField(ValueA)) {
    ValueA = TheHelper_.getAsAlloca(ValueA);
    IndexPartitionA = TheHelper_.getAsAlloca(IndexPartitionA);
    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ValueA,
      IndexSpaceA,
      IndexPartitionA,
      PartInfoA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition_from_field",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------

  return IndexPartA;
    
}

//==============================================================================
// create a range
//==============================================================================
AllocaInst* LegionTasker::createPartition(
    Module & TheModule,
    Value* IndexSpaceA,
    Value* Color)
{

  auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  auto IndexPartA = TheHelper_.createEntryBlockAlloca(IndexPartitionType_);
    
  IndexSpaceA = TheHelper_.getAsAlloca(IndexSpaceA);

  //------------------------------------
  if (isRange(Color)) {
    auto ColorA = TheHelper_.getAsAlloca(Color);

    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ColorA,
      IndexSpaceA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------
  else if (librt::DopeVector::isDopeVector(Color)) {
    auto ColorA = TheHelper_.getAsAlloca(Color);
  
    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ColorA,
      IndexSpaceA,
      IndexPartA,
      llvmValue<bool>(getContext(), true)
    };
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition_from_array",
        VoidType_,
        FunArgVs);
  }
  //------------------------------------
  else {
    auto ColorV = TheHelper_.getAsValue(Color);

    std::vector<Value*> FunArgVs = {
      RuntimeA,
      ContextA,
      ColorV,
      IndexSpaceA,
      IndexPartA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_partition_from_size",
        VoidType_,
        FunArgVs);
  }
  
  return IndexPartA;

}


//==============================================================================
// destroey a field
//==============================================================================
void LegionTasker::destroyRange(Module &TheModule, Value* RangeV)
{
  if (isInsideTask()) {
    const auto & LegionE = getCurrentTask();
    const auto & ContextA = LegionE.ContextAlloca;
    const auto & RuntimeA = LegionE.RuntimeAlloca;

    Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
    
    std::vector<Value*> FunArgVs = {RuntimeA, ContextA, IndexSpaceA};
    
    TheHelper_.callFunction(
        TheModule,
        "contra_legion_index_space_destroy",
        VoidType_,
        FunArgVs);
  }
}

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeStart(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 0); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeEnd(Value* RangeV)
{
  Value* EndV = TheHelper_.extractValue(RangeV, 1);
  auto OneC = llvmValue<int_t>(getContext(), 1);
  return getBuilder().CreateSub(EndV, OneC);
}


//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeEndPlusOne(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 1); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* LegionTasker::getRangeStep(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 2); }

//==============================================================================
// get a range size
//==============================================================================
llvm::Value* LegionTasker::getRangeSize(Value* RangeV)
{
  auto StartV = TheHelper_.extractValue(RangeV, 0);
  auto EndV = TheHelper_.extractValue(RangeV, 1);
  return getBuilder().CreateSub(EndV, StartV);
}

//==============================================================================
// get a range value
//==============================================================================
llvm::Value* LegionTasker::loadRangeValue(
    Value* RangeA,
    Value* IndexV)
{
  auto StartV = TheHelper_.extractValue(RangeA, 0); 
  IndexV = TheHelper_.getAsValue(IndexV);
  return getBuilder().CreateAdd(StartV, IndexV);
}


//==============================================================================
// Is this an accessor type
//==============================================================================
bool LegionTasker::isAccessor(Type* AccessorT) const
{
  return (AccessorT == AccessorDataType_);
}

bool LegionTasker::isAccessor(Value* AccessorV) const
{
  auto AccessorT = AccessorV->getType();
  if (auto AccessorA = dyn_cast<AllocaInst>(AccessorV))
    AccessorT = AccessorA->getAllocatedType();
  return isAccessor(AccessorT);
}

//==============================================================================
// Store a value into an accessor
//==============================================================================
void LegionTasker::storeAccessor(
    Module & TheModule,
    Value* ValueV,
    Value* AccessorV,
    Value* IndexV)
{
  auto ValueA = TheHelper_.getAsAlloca(ValueV);

  Value* AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  std::vector<Value*> FunArgVs = { AccessorA, ValueA };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(getContext(), 0) );
  }
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_accessor_write",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Load a value from an accessor
//==============================================================================
Value* LegionTasker::loadAccessor(
    Module & TheModule, 
    Type * ValueT,
    Value* AccessorV,
    Value* IndexV)
{
  auto AccessorA = TheHelper_.getAsAlloca(AccessorV);
    
  auto ValueA = TheHelper_.createEntryBlockAlloca(ValueT);

  std::vector<Value*> FunArgVs = { AccessorA, ValueA };
  
  if (IndexV) {
    FunArgVs.emplace_back( TheHelper_.getAsValue(IndexV) );
  }
  else {
    FunArgVs.emplace_back( llvmValue<int_t>(getContext(), 0) );
  }

  TheHelper_.callFunction(
      TheModule,
      "contra_legion_accessor_read",
      VoidType_,
      FunArgVs);

  return ValueA;
}

//==============================================================================
// destroey an accessor
//==============================================================================
void LegionTasker::destroyAccessor(
    Module &TheModule,
    Value* AccessorA)
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  std::vector<Value*> FunArgVs = {RuntimeA, ContextA, AccessorA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_accessor_destroy",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// Is this an range type
//==============================================================================
bool LegionTasker::isPartition(Type* PartT) const
{
  return (PartT == IndexPartitionType_);
}

bool LegionTasker::isPartition(Value* PartV) const
{
  auto PartT = PartV->getType();
  if (auto PartA = dyn_cast<AllocaInst>(PartV))
    PartT = PartA->getAllocatedType();
  return isPartition(PartT);
}

//==============================================================================
// destroey an accessor
//==============================================================================
void LegionTasker::destroyPartition(
    Module &TheModule,
    Value* PartitionA)
{
  const auto & LegionE = getCurrentTask();
  const auto & ContextA = LegionE.ContextAlloca;
  const auto & RuntimeA = LegionE.RuntimeAlloca;

  std::vector<Value*> FunArgVs = {RuntimeA, ContextA, PartitionA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_partition_destroy",
      VoidType_,
      FunArgVs);
}

//==============================================================================
// create reduction funcction
//==============================================================================
Function* LegionTasker::createReductionFunction(
    Module & TheModule,
    const std::string & FunN,
    const std::string & OpN,
    const std::vector<std::size_t> & DataSizes,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{
  std::vector<Type*> ArgTs = {
    VoidPtrType_,
    VoidPtrType_,
    OffType_,
    OffType_,
    SizeType_,
    BoolType_
  };
  FunctionType* FunT = FunctionType::get(VoidType_, ArgTs, false);
  auto FunF = Function::Create(
      FunT,
      Function::ExternalLinkage,
      FunN,
      TheModule);

  auto BB = BasicBlock::Create(getContext(), "entry", FunF);
  getBuilder().SetInsertPoint(BB);

  unsigned i=0;
  std::vector<AllocaInst*> ArgAs(ArgTs.size());
  for (auto &Arg : FunF->args()) {
    ArgAs[i] = TheHelper_.createEntryBlockAlloca(ArgTs[i]);
    getBuilder().CreateStore(&Arg, ArgAs[i]);
    ++i;
  }

  auto CounterA = TheHelper_.createEntryBlockAlloca(SizeType_);
  auto InnerOffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
  auto OuterOffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
  auto ZeroC = llvmValue(getContext(), SizeType_, 0);
  getBuilder().CreateStore(ZeroC, CounterA);
  getBuilder().CreateStore(ZeroC, OuterOffsetA);

  auto BeforeBB = BasicBlock::Create(getContext(), "beforeloop", FunF);
  auto LoopBB =   BasicBlock::Create(getContext(), "loop", FunF);
  auto IncrBB =   BasicBlock::Create(getContext(), "incr", FunF);
  auto AfterBB =  BasicBlock::Create(getContext(), "afterloop", FunF);

  getBuilder().CreateBr(BeforeBB);
  getBuilder().SetInsertPoint(BeforeBB);

  auto CounterV = TheHelper_.load(CounterA);
  auto EndV = TheHelper_.load(ArgAs[4]);
  auto CondV = getBuilder().CreateICmpSLT(CounterV, EndV, "loopcond");
  getBuilder().CreateCondBr(CondV, LoopBB, AfterBB);

  getBuilder().SetInsertPoint(LoopBB);

  auto OuterOffsetV = TheHelper_.load(OuterOffsetA);
  getBuilder().CreateStore( OuterOffsetV, InnerOffsetA);

  for (unsigned i=0; i<VarTs.size(); ++i) {
    Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
    Value* RhsPtrV = TheHelper_.load(ArgAs[1]);
    auto OffsetV = TheHelper_.load(InnerOffsetA);
    LhsPtrV = getBuilder().CreateGEP(ArgTs[0], LhsPtrV, OffsetV);
    RhsPtrV = getBuilder().CreateGEP(ArgTs[1], RhsPtrV, OffsetV);
    auto VarT = VarTs[i];
    auto VarPtrT = VarT->getPointerTo();
    LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarPtrT);
    RhsPtrV = TheHelper_.createBitCast(RhsPtrV, VarPtrT);
    std::string TypeN;
    if (VarT->isFloatingPointTy())
      TypeN = "real";
    else
      TypeN = "int";
    std::string ReduceN;
    if (ReduceTypes[i] == ReductionType::Add)
      ReduceN = "sum";
    else if (ReduceTypes[i] == ReductionType::Sub)
      ReduceN = "sub";
    else if (ReduceTypes[i] == ReductionType::Mult)
      ReduceN = "mul";
    else if (ReduceTypes[i] == ReductionType::Div)
      ReduceN = "div";
    else if (ReduceTypes[i] == ReductionType::Min)
      ReduceN = "min";
    else if (ReduceTypes[i] == ReductionType::Max)
      ReduceN = "max";
    else {
      std::cerr << "Unsupported reduction op." << std::endl;;
      abort();
    }
    auto FinalN = "contra_" + ReduceN + "_" + OpN + "_" + TypeN;
    TheHelper_.callFunction(
        TheModule,
        FinalN,
        VoidType_,
        {LhsPtrV, RhsPtrV, ArgAs[5]});
    auto SizeC = llvmValue(getContext(), SizeType_, DataSizes[i]);
    TheHelper_.increment( InnerOffsetA, SizeC );
  }
  
  
  getBuilder().CreateBr(IncrBB);
  getBuilder().SetInsertPoint(IncrBB);
  
  auto StrideV = TheHelper_.load(ArgAs[2]);
  TheHelper_.increment( OuterOffsetA, StrideV );
  
  auto OneC = llvmValue(getContext(), SizeType_, 1);
  TheHelper_.increment( CounterA, OneC );
  getBuilder().CreateBr(BeforeBB);
  getBuilder().SetInsertPoint(AfterBB);
  
  getBuilder().CreateRetVoid();

  return FunF;
}

//==============================================================================
// create a reduction op
//==============================================================================
std::unique_ptr<AbstractReduceInfo> LegionTasker::createReductionOp(
    Module &TheModule,
    const std::string & ReductionN,
    const std::vector<Type*> & VarTs,
    const std::vector<ReductionType> & ReduceTypes)
{
  // generate id
  auto RedOpId = makeReductionId();

  // get var types

  // get data size
  std::size_t DataSize = 0;
  std::vector<std::size_t> DataSizes;
  for (auto VarT : VarTs) {
    DataSizes.emplace_back( TheHelper_.getTypeSizeInBits(TheModule, VarT)/8 );
    DataSize += DataSizes.back();
  }


  //----------------------------------------------------------------------------
  // create apply
  auto ApplyF = createReductionFunction(
      TheModule,
      ReductionN + "apply",
      "apply",
      DataSizes,
      VarTs,
      ReduceTypes);
  
  //----------------------------------------------------------------------------
  // create fold
  auto FoldF = createReductionFunction(
      TheModule,
      ReductionN + "fold",
      "fold",
      DataSizes,
      VarTs,
      ReduceTypes);

  //----------------------------------------------------------------------------
  // create init
  Function *InitF;
  {

    std::string InitN = ReductionN + "init";
 
    std::vector<Type*> ArgTs = {
      VoidPtrType_,
      SizeType_
    };
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
    auto CounterA = TheHelper_.createEntryBlockAlloca(SizeType_);
    auto OffsetA = TheHelper_.createEntryBlockAlloca(SizeType_);
    getBuilder().CreateStore(ZeroC, CounterA);
    getBuilder().CreateStore(ZeroC, OffsetA);
  
    auto BeforeBB = BasicBlock::Create(getContext(), "beforeloop", InitF);
    auto LoopBB =   BasicBlock::Create(getContext(), "loop", InitF);
    auto IncrBB =   BasicBlock::Create(getContext(), "incr", InitF);
    auto AfterBB =  BasicBlock::Create(getContext(), "afterloop", InitF);
  
    getBuilder().CreateBr(BeforeBB);
    getBuilder().SetInsertPoint(BeforeBB);

    auto CounterV = TheHelper_.load(CounterA);
    auto EndV = TheHelper_.load(ArgAs[1]);
    auto CondV = getBuilder().CreateICmpSLT(CounterV, EndV, "loopcond");
    getBuilder().CreateCondBr(CondV, LoopBB, AfterBB);

    getBuilder().SetInsertPoint(LoopBB);


    for (unsigned i=0; i<VarTs.size(); ++i) {
      Value* LhsPtrV = TheHelper_.load(ArgAs[0]);
      auto OffsetV = TheHelper_.load(OffsetA);
      LhsPtrV = getBuilder().CreateGEP(ArgTs[0], LhsPtrV, OffsetV);
      LhsPtrV = TheHelper_.createBitCast(LhsPtrV, VarTs[i]->getPointerTo());
      auto VarT = VarTs[i];
      auto InitC = initReduce(VarT, ReduceTypes[i]);
      getBuilder().CreateStore(InitC, LhsPtrV);
      auto SizeC = llvmValue(getContext(), SizeType_, DataSizes[i]);
      TheHelper_.increment( OffsetA, SizeC );
    }
    
    getBuilder().CreateBr(IncrBB);
    getBuilder().SetInsertPoint(IncrBB);
    
    auto OneC = llvmValue(getContext(), SizeType_, 1);
    TheHelper_.increment( CounterA, OneC );
    getBuilder().CreateBr(BeforeBB);
    getBuilder().SetInsertPoint(AfterBB);
    
    getBuilder().CreateRetVoid();
  }
  

  //----------------------------------------------------------------------------
  // create reduction
  return 
    std::make_unique<LegionReduceInfo>(RedOpId, ApplyF, FoldF, InitF, DataSize);

}

//==============================================================================
// call a reduction creation op
//==============================================================================
void LegionTasker::registerReductionOp(
    Module &TheModule,
    const AbstractReduceInfo * ParentReduceOp)
{
  auto ReduceOp = dynamic_cast<const LegionReduceInfo*>(ParentReduceOp);

  auto DataSizeC = llvmValue(getContext(), SizeType_, ReduceOp->getDataSize());
  auto IdC = llvmValue<legion_reduction_op_id_t>(getContext(), ReduceOp->getId());
  
  auto ApplyT = ReduceOp->getApplyType();
  const auto & ApplyN = ReduceOp->getApplyName();
  auto ApplyF = TheModule.getOrInsertFunction(ApplyN, ApplyT).getCallee();

  auto FoldT = ReduceOp->getFoldType();
  const auto & FoldN = ReduceOp->getFoldName();
  auto FoldF = TheModule.getOrInsertFunction(FoldN, FoldT).getCallee();
  
  auto InitT = ReduceOp->getInitType();
  const auto & InitN = ReduceOp->getInitName();
  auto InitF = TheModule.getOrInsertFunction(InitN, InitT).getCallee();
        
  TheHelper_.callFunction(
      TheModule,
      "contra_legion_create_reduction",
      VoidType_,
      {IdC, ApplyF, FoldF, InitF, DataSizeC});
  
}

}
