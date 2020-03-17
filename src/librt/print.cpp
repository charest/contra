#include "llvm_includes.hpp"
#include "print.hpp"

#include "contra/config.hpp"
#include "contra/context.hpp"
#include "contra/symbols.hpp"

#include <cstdarg>

extern "C" {

//==============================================================================
/// generic c print statement
//==============================================================================
void print(const char *format, ...)
{

   va_list arg;

   va_start (arg, format);
   vfprintf (stdout, format, arg);
   va_end (arg);

}

} // extern

namespace librt {

using namespace contra;
using namespace llvm;

//==============================================================================
// Installs the print function
//==============================================================================
const std::string Print::Name = "print";

Function * Print::install(LLVMContext & TheContext, Module & TheModule)
{
  auto PrintType = FunctionType::get(
      Type::getVoidTy(TheContext),
      llvmVoidPointerType(TheContext),
      true /* var args */ );

  //auto PrintFun = TheModule.getOrInsertFunction("print", PrintType);
  auto PrintFun = Function::Create(PrintType, Function::ExternalLinkage,
      Print::Name, TheModule);
  return PrintFun;
}


std::shared_ptr<contra::FunctionDef> Print::check()
{
  std::vector<VariableType> Args;
  Args.emplace_back( Context::StrType );
  return std::make_shared<BuiltInFunction>(Print::Name, 
      VariableType(Context::VoidType), Args, true);
}

}
