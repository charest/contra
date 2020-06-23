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
  launch(TheModule, *TopLevelTask_, {});
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
  abort();
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
