#include "config.hpp"
#include "llvm_includes.hpp"

#include <cstdlib>

extern "C" {

/// simple dopevector type
struct dopevector_t {
  void * data = nullptr;
  int_t size = 0;
};


//==============================================================================
/// memory allocation
//==============================================================================
dopevector_t allocate(int_t size)
{
  dopevector_t dv;
  dv.data = malloc(size);
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

//==============================================================================
// Create the dopevector type 
//==============================================================================
Type * createDopeVectorType(LLVMContext & TheContext)
{
  auto DopeVectorType = StructType::create( TheContext, "dopevector_t" );
  auto VoidPointerType = llvmVoidPointerType(TheContext);
  auto IntType = llvmIntegerType(TheContext);

  std::vector<Type*> members{ VoidPointerType, IntType}; 
  DopeVectorType->setBody( members );

  return DopeVectorType;
}

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *installAllocate(LLVMContext & TheContext, Module & TheModule)
{
  auto DopeVectorType = createDopeVectorType(TheContext);
  auto IntType = llvmIntegerType(TheContext);

  std::vector<Type*> Args = {IntType};
  auto AllocateType = FunctionType::get( DopeVectorType, Args, false );

  auto AllocateFun = Function::Create(AllocateType, Function::ExternalLinkage,
      "allocate", TheModule);
  return AllocateFun;
}

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *installDeAllocate(LLVMContext & TheContext, Module & TheModule)
{
  auto DopeVectorType = createDopeVectorType(TheContext);
  auto VoidType = Type::getVoidTy(TheContext);

  std::vector<Type*> Args = {DopeVectorType};
  auto DeAllocateType = FunctionType::get( VoidType, Args, false );

  auto DeAllocateFun = Function::Create(DeAllocateType, Function::ExternalLinkage,
      "deallocate", TheModule);
  
  return DeAllocateFun;
}

}
