#include "dopevector.hpp"
#include "llvm_includes.hpp"

#include "llvm_utils.hpp"

#include <cstdlib>
#include <iostream>

extern "C" {

/// simple dopevector type
struct dopevector_t {
  void * data = nullptr;
  int_t size = 0;
};


//==============================================================================
/// memory allocation
//==============================================================================
dopevector_t allocate(int_t size, int_t data_size)
{
  dopevector_t dv;
  dv.data = malloc(size*data_size);
  dv.size = size;
  return dv;
}

//==============================================================================
/// memory deallocation
//==============================================================================
void deallocate(dopevector_t dv)
{
  free(dv.data);
}

} // extern

namespace librt {

using namespace llvm;

Type* DopeVector::DopeVectorType = nullptr;
const std::string Allocate::Name = "allocate";
const std::string DeAllocate::Name = "deallocate";

//==============================================================================
// Create the dopevector type 
//==============================================================================
Type * createDopeVectorType(LLVMContext & TheContext)
{
  auto DopeVectorType = StructType::create( TheContext, "dopevector_t" );
  auto VoidPointerType = llvmType<void*>(TheContext);
  auto IntType = llvmType<int_t>(TheContext);

  std::vector<Type*> members{ VoidPointerType, IntType }; 
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

  std::vector<Type*> Args = {IntType, IntType};
  auto AllocateType = FunctionType::get( DopeVectorType, Args, false );

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

  std::vector<Type*> Args = {DopeVectorType};
  auto DeAllocateType = FunctionType::get( VoidType, Args, false );

  auto DeAllocateFun = Function::Create(DeAllocateType, Function::InternalLinkage,
      DeAllocate::Name, TheModule);
  
  return DeAllocateFun;
}

}
