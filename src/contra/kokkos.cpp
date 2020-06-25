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
  FieldDataType_ = createFieldDataType();
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
void KokkosTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{

  auto ArgcV = llvmValue(TheContext_, Int32Type_, Argc);

  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argc; ++i)
    ArgVs.emplace_back( llvmString(TheContext_, TheModule, Argv[i]) );

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  auto ArgvV = llvmArray(TheContext_, TheModule, ArgVs, {ZeroC, ZeroC});

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV };
  TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_runtime_start",
      Int32Type_,
      StartArgVs,
      "start");
  
  launch(TheModule, *TopLevelTask_);
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
