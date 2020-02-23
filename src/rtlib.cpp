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

/// generic c print statement
extern "C" DLLEXPORT double print(const char *format, ...)
{

   va_list arg;
   int done;

   va_start (arg, format);
   done = vfprintf (stdout, format, arg);
   va_end (arg);

   return done;
}

namespace contra {

using namespace llvm;

//==============================================================================
// Installs the print function
//==============================================================================
Function *installPrint(LLVMContext & TheContext, Module & TheModule)
{
  auto PrintType = FunctionType::get(
      Type::getDoubleTy(TheContext),
      PointerType::get(Type::getInt8Ty(TheContext), 0),
      true /* var args */ );

  //auto PrintFun = TheModule.getOrInsertFunction("print", PrintType);
  auto PrintFun = Function::Create(PrintType, Function::ExternalLinkage,
      "print", TheModule);
  return PrintFun;
}

//==============================================================================
// Installs the unary minus
//==============================================================================
Function *installUnaryNegate(LLVMContext & TheContext, Module & TheModule)
{
  abort();
  return nullptr;
}

//==============================================================================
// install the library functions available by default
//==============================================================================
std::map<std::string, RunTimeLib::InstallFunctionPointer>
  RunTimeLib::InstallMap = {
    {"print",installPrint},
    {"unary-",installUnaryNegate}
  };

}
