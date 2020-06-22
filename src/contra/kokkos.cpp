#include "kokkos.hpp"

#include "errors.hpp"
#include "kokkos_rt.hpp"

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
KokkosTasker::KokkosTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{
  IndexSpaceDataType_ = createIndexSpaceDataType();
  FieldDataType_ = createFieldDataType();
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * KokkosTasker::createIndexSpaceDataType()
{
  std::vector<Type*> members = { IntType_, IntType_, IntType_ };
  auto NewType = StructType::create( TheContext_, members, "contra_kokkos_index_space_t" );
  return NewType;
}

//==============================================================================
// Create the field data type
//==============================================================================
StructType * KokkosTasker::createFieldDataType()
{
  std::vector<Type*> members = { Int32Type_, VoidPtrType_ };
  auto NewType = StructType::create( TheContext_, members, "contra_kokkos_field_t" );
  return NewType;
}


//==============================================================================
// start runtime
//==============================================================================
Value* KokkosTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{

  auto ArgcV = llvmValue(TheContext_, Int32Type_, Argc);

  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argc; ++i)
    ArgVs.emplace_back( llvmString(TheContext_, TheModule, Argv[i]) );

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  auto ArgvV = llvmArray(TheContext_, TheModule, ArgVs, {ZeroC, ZeroC});

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV };
  auto RetI = TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_runtime_start",
      Int32Type_,
      StartArgVs,
      "start");
  
  launch(TheModule, *TopLevelTask_, {});

  return RetI;
}

//==============================================================================
// stop runtime
//=============================================================================
void KokkosTasker::stopRuntime(Module &TheModule)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_runtime_stop",
      VoidType_,
      {});
}

//==============================================================================
// Create the function wrapper
//==============================================================================
KokkosTasker::PreambleResult KokkosTasker::taskPreamble(
    Module &TheModule,
    const std::string & TaskName,
    const std::vector<std::string> & TaskArgNs,
    const std::vector<Type*> & TaskArgTs)
{
  abort();
}

//==============================================================================
// Is this an range type
//==============================================================================
bool KokkosTasker::isRange(Type* RangeT) const
{
  return (RangeT == IndexSpaceDataType_);
}

bool KokkosTasker::isRange(Value* RangeA) const
{
  auto RangeT = RangeA->getType();
  if (isa<AllocaInst>(RangeA)) RangeT = RangeT->getPointerElementType();
  return isRange(RangeT);
}


//==============================================================================
// create a range
//==============================================================================
AllocaInst* KokkosTasker::createRange(
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

  TheHelper_.insertValue(IndexSpaceA, StartV, 0);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  EndV = Builder_.CreateAdd(EndV, OneC);
  TheHelper_.insertValue(IndexSpaceA, EndV, 1);
  if (!StepV) StepV = OneC;
  TheHelper_.insertValue(IndexSpaceA, StepV, 2);
  
  return IndexSpaceA;

}

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* KokkosTasker::getRangeStart(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 0); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* KokkosTasker::getRangeEnd(Value* RangeV)
{
  Value* EndV = TheHelper_.extractValue(RangeV, 1);
  auto OneC = llvmValue<int_t>(TheContext_, 1);
  return Builder_.CreateSub(EndV, OneC);
}


//==============================================================================
// get a range start
//==============================================================================
llvm::Value* KokkosTasker::getRangeEndPlusOne(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 1); }

//==============================================================================
// get a range start
//==============================================================================
llvm::Value* KokkosTasker::getRangeStep(Value* RangeV)
{ return TheHelper_.extractValue(RangeV, 2); }

//==============================================================================
// get a range size
//==============================================================================
llvm::Value* KokkosTasker::getRangeSize(Value* RangeV)
{
  auto StartV = TheHelper_.extractValue(RangeV, 0);
  auto EndV = TheHelper_.extractValue(RangeV, 1);
  return Builder_.CreateSub(EndV, StartV);
}

//==============================================================================
// get a range value
//==============================================================================
llvm::Value* KokkosTasker::loadRangeValue(
    Value* RangeA,
    Value* IndexV)
{
  auto StartV = TheHelper_.extractValue(RangeA, 0); 
  IndexV = TheHelper_.getAsValue(IndexV);
  return Builder_.CreateAdd(StartV, IndexV);
}

//==============================================================================
// Is this a field type
//==============================================================================
bool KokkosTasker::isField(Value* FieldA) const
{
  auto FieldT = FieldA->getType();
  if (isa<AllocaInst>(FieldA)) FieldT = FieldT->getPointerElementType();
  return (FieldT == FieldDataType_);
}


//==============================================================================
// Create a legion field
//==============================================================================
void KokkosTasker::createField(
    Module & TheModule,
    Value* FieldA,
    const std::string & VarN,
    Type* VarT,
    Value* RangeV,
    Value* VarV)
{
  auto NameV = llvmString(TheContext_, TheModule, VarN);

  Value* DataTypeV;
  if (VarT == IntType_)
    DataTypeV = llvmValue<int>(TheContext_, KokkosFieldType::Integer);
  else if (VarT == RealType_)
    DataTypeV = llvmValue<int>(TheContext_, KokkosFieldType::Real);
  else {
    std::string str;
    raw_string_ostream out(str);
    VarT->print(out);
    THROW_CONTRA_ERROR("Unknown field type: " << str);
  }

  if (VarV)
    VarV = TheHelper_.getAsAlloca(VarV);
  else
    VarV = Constant::getNullValue(VoidPtrType_);
    
  Value* IndexSpaceA = TheHelper_.getAsAlloca(RangeV);
  
  std::vector<Value*> FunArgVs = {
    NameV,
    DataTypeV, 
    VarV,
    IndexSpaceA,
    FieldA};
  
  TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_field_create",
      VoidType_,
      FunArgVs);
    
}

//==============================================================================
// destroey a field
//==============================================================================
void KokkosTasker::destroyField(Module &TheModule, Value* FieldA)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_field_destroy",
      VoidType_,
      {FieldA});
}

}
