#include "llvm_includes.hpp"

#include <cstdarg>

extern "C" {

//==============================================================================
/// generic c print statement
//==============================================================================
void print(const char *format, ...)
{

   va_list arg;
   int done;

   va_start (arg, format);
   done = vfprintf (stdout, format, arg);
   va_end (arg);

}

} // extern

namespace librt {

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



}
