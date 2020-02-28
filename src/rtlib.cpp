#include "rtlib.hpp"

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <stdio.h>
#include <stdarg.h>

//==============================================================================
// "Library" functions that can be "extern'd" from user code.
//==============================================================================

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {

/// generic c print statement
DLLEXPORT void print(const char *format, ...)
{

   va_list arg;
   int done;

   va_start (arg, format);
   done = vfprintf (stdout, format, arg);
   va_end (arg);

}

struct dopevector_t {
  void * data = nullptr;
  std::uint64_t size = 0;
};

/// memory allocation
DLLEXPORT dopevector_t allocate(std::uint64_t size)
{
  dopevector_t dv;
  dv.data = malloc(size);
  dv.size = size;
  return dv;
}

DLLEXPORT void deallocate(dopevector_t dv)
{
  free(dv.data);
}

} // extern

namespace contra {

using namespace llvm;

//==============================================================================
// Installs the print function
//==============================================================================
Function *installPrint(LLVMContext & TheContext, Module & TheModule)
{
  auto PrintType = FunctionType::get(
      Type::getVoidTy(TheContext),
      PointerType::get(Type::getInt8Ty(TheContext), 0),
      true /* var args */ );

  //auto PrintFun = TheModule.getOrInsertFunction("print", PrintType);
  auto PrintFun = Function::Create(PrintType, Function::ExternalLinkage,
      "print", TheModule);
  return PrintFun;
}

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *installAllocate(LLVMContext & TheContext, Module & TheModule)
{
  auto DopeVectorType = StructType::create( TheContext, "dopevector_t" );
  auto VoidPointerType = PointerType::get(Type::getInt8Ty(TheContext), 0);
  auto Int64Type = Type::getInt64Ty(TheContext);

  std::vector<Type*> members{ VoidPointerType, Int64Type}; 
  DopeVectorType->setBody( members );
  //DopeVectorType->print(outs()); outs() << "\n";

  std::vector<Type*> Args = {Int64Type};
  auto AllocateType = FunctionType::get( DopeVectorType, Args, false );

  auto AllocateFun = Function::Create(AllocateType, Function::ExternalLinkage,
      "allocate", TheModule);
  //AllocateFun->print(outs()); outs() << "\n";
  return AllocateFun;
}

//==============================================================================
// Installs the Allocate deallocate function
//==============================================================================
Function *installDeAllocate(LLVMContext & TheContext, Module & TheModule)
{
  auto DopeVectorType = StructType::create( TheContext, "dopevector_t" );
  auto VoidPointerType = PointerType::get(Type::getInt8Ty(TheContext), 0);
  auto Int64Type = Type::getInt64Ty(TheContext);

  std::vector<Type*> members{ VoidPointerType, Int64Type}; 
  DopeVectorType->setBody( members );
  //DopeVectorType->print(outs()); outs() << "\n";

  auto VoidType = Type::getVoidTy(TheContext);

  std::vector<Type*> Args = {DopeVectorType};
  auto DeAllocateType = FunctionType::get( VoidType, Args, false );

  auto DeAllocateFun = Function::Create(DeAllocateType, Function::ExternalLinkage,
      "deallocate", TheModule);
  //AllocateFun->print(outs()); outs() << "\n";
  return DeAllocateFun;
}


//==============================================================================
// install the library functions available by default
//==============================================================================
std::map<std::string, RunTimeLib::InstallFunctionPointer>
  RunTimeLib::InstallMap = {
    {"print",installPrint},
    {"allocate",installAllocate},
    {"deallocate",installDeAllocate},
  };

}
