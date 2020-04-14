#include "dopevector.hpp"
#include "llvm_includes.hpp"

#include "utils/llvm_utils.hpp"

#include <cstdlib>
#include <iostream>

extern "C" {

/// simple dopevector type
struct dopevector_t {
  void * data = nullptr;
  int_t size = 0;
  int_t capacity = 0;
  int_t data_size = 0;
};


//==============================================================================
/// memory allocation
//==============================================================================
void allocate(int_t size, int_t data_size, dopevector_t * dv)
{
  dv->data = malloc(size*data_size);
  dv->size = size;
  dv->capacity = size;
  dv->data_size = data_size;
}

//==============================================================================
/// memory deallocation
//==============================================================================
void deallocate(dopevector_t * dv)
{
  free(dv->data);
  dv->size = 0;
  dv->capacity = 0;
  dv->data_size = 0;
}

//==============================================================================
/// copy
//==============================================================================
void copy(dopevector_t * src, dopevector_t * tgt)
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

using namespace llvm;
using namespace utils;

Type* DopeVector::DopeVectorType = nullptr;
const std::string Allocate::Name = "allocate";
const std::string DeAllocate::Name = "deallocate";
const std::string Copy::Name = "copy";

//==============================================================================
// Create the dopevector type 
//==============================================================================
Type * createDopeVectorType(LLVMContext & TheContext)
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

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *Allocate::install(LLVMContext & TheContext, Module & TheModule)
{
  auto IntType = llvmType<int_t>(TheContext);
  auto VoidType = Type::getVoidTy(TheContext);

  std::vector<Type*> Args = {IntType, IntType, DopeVectorType->getPointerTo()};
  auto AllocateType = FunctionType::get( VoidType, Args, false );

  auto AllocateFun = Function::Create(AllocateType, Function::InternalLinkage,
      Allocate::Name, TheModule);
  return AllocateFun;
}

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *DeAllocate::install(LLVMContext & TheContext, Module & TheModule)
{
  auto VoidType = Type::getVoidTy(TheContext);

  std::vector<Type*> Args = {DopeVectorType->getPointerTo()};
  auto DeAllocateType = FunctionType::get( VoidType, Args, false );

  auto DeAllocateFun = Function::Create(DeAllocateType, Function::InternalLinkage,
      DeAllocate::Name, TheModule);
  
  return DeAllocateFun;
}

//==============================================================================
// Installs the copy function
//==============================================================================
Function *Copy::install(LLVMContext & TheContext, Module & TheModule)
{
  auto VoidType = Type::getVoidTy(TheContext);

  auto DopeVectorPtrType = DopeVectorType->getPointerTo();
  std::vector<Type*> Args = {DopeVectorPtrType, DopeVectorPtrType};
  auto FunT = FunctionType::get( VoidType, Args, false );

  auto FunF = Function::Create(FunT, Function::InternalLinkage, Copy::Name, TheModule);
  
  return FunF;
}

}
