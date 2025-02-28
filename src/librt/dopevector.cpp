#include "dopevector.hpp"
#include "llvm_includes.hpp"

#include "contra/errors.hpp"
#include "contra/symbols.hpp"
#include "utils/llvm_utils.hpp"

#include <cstdlib>
#include <iostream>

extern "C" {

//==============================================================================
/// memory allocation
//==============================================================================
void dopevector_allocate(int_t size, int_t data_size, dopevector_t * dv)
{
  dv->data = malloc(size*data_size);
  dv->size = size;
  dv->capacity = size;
  dv->data_size = data_size;
}

//==============================================================================
/// memory deallocation
//==============================================================================
void dopevector_deallocate(dopevector_t * dv)
{
  free(dv->data);
  dv->size = 0;
  dv->capacity = 0;
  dv->data_size = 0;
}

//==============================================================================
/// copy
//==============================================================================
void dopevector_copy(dopevector_t * src, dopevector_t * tgt)
{
  int_t len = src->size*src->data_size;
  if (tgt->capacity < src->size) {
    free(tgt->data);
    tgt->data = malloc(len);
    tgt->capacity = src->size;
  }
  tgt->size = src->size;
  memcpy(tgt->data, src->data, len);
}

} // extern

namespace librt {

using namespace contra;
using namespace llvm;
using namespace utils;

StructType* DopeVector::DopeVectorType = nullptr;
const std::string DopeVectorAllocate::Name = "dopevector_allocate";
const std::string DopeVectorDeAllocate::Name = "dopevector_deallocate";
const std::string DopeVectorCopy::Name = "dopevector_copy";

//==============================================================================
// Create the dopevector type 
//==============================================================================
StructType * createDopeVectorType(LLVMContext & TheContext)
{
  auto DopeVectorType = StructType::create( TheContext, "dopevector_t" );
  auto VoidPointerType = llvmType<void*>(TheContext);
  auto IntType = llvmType<int_t>(TheContext);

  std::vector<Type*> members{ VoidPointerType, IntType, IntType, IntType }; 
  DopeVectorType->setBody( members );

  return DopeVectorType;
}

//==============================================================================
// Sets up whatever is needed for allocate
//==============================================================================
void DopeVector::setup(LLVMContext & TheContext)
{ 
  if (!DopeVectorType)
    DopeVectorType = createDopeVectorType(TheContext);
}
  
bool DopeVector::isDopeVector(Type* Ty) 
{ return DopeVectorType == Ty; }
  
bool DopeVector::isDopeVector(Value* V)
{
  auto Ty = V->getType();
  if (isa<AllocaInst>(V)) Ty = cast<AllocaInst>(V)->getAllocatedType();
  return isDopeVector(Ty);
}

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *DopeVectorAllocate::install(LLVMContext & TheContext, Module & TheModule)
{
  auto IntType = llvmType<int_t>(TheContext);
  auto VoidType = Type::getVoidTy(TheContext);

  std::vector<Type*> Args = {IntType, IntType, DopeVectorType->getPointerTo()};
  auto AllocateType = FunctionType::get( VoidType, Args, false );

  auto AllocateFun = Function::Create(AllocateType, Function::InternalLinkage,
      DopeVectorAllocate::Name, TheModule);
  return AllocateFun;
}

std::unique_ptr<FunctionDef> DopeVectorAllocate::check()
{ return std::unique_ptr<BuiltInFunction>(nullptr); }

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *DopeVectorDeAllocate::install(LLVMContext & TheContext, Module & TheModule)
{
  auto VoidType = Type::getVoidTy(TheContext);

  std::vector<Type*> Args = {DopeVectorType->getPointerTo()};
  auto DeAllocateType = FunctionType::get( VoidType, Args, false );

  auto DeAllocateFun = Function::Create(DeAllocateType, Function::InternalLinkage,
      DopeVectorDeAllocate::Name, TheModule);
  
  return DeAllocateFun;
}

std::unique_ptr<FunctionDef> DopeVectorDeAllocate::check()
{ return std::unique_ptr<BuiltInFunction>(nullptr); }

//==============================================================================
// Installs the copy function
//==============================================================================
Function *DopeVectorCopy::install(LLVMContext & TheContext, Module & TheModule)
{
  auto VoidType = Type::getVoidTy(TheContext);

  auto DopeVectorPtrType = DopeVectorType->getPointerTo();
  std::vector<Type*> Args = {DopeVectorPtrType, DopeVectorPtrType};
  auto FunT = FunctionType::get( VoidType, Args, false );

  auto FunF = Function::Create(FunT, Function::InternalLinkage, DopeVectorCopy::Name, TheModule);
  
  return FunF;
}

std::unique_ptr<FunctionDef> DopeVectorCopy::check()
{ return std::unique_ptr<BuiltInFunction>(nullptr); }

}
