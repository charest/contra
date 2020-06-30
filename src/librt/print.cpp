#include "llvm_includes.hpp"
#include "print.hpp"

#include "config.hpp"
#include "contra/context.hpp"
#include "contra/symbols.hpp"
#include "utils/llvm_utils.hpp"

#include <cstdarg>

extern "C" {

//==============================================================================
/// generic c print statement
//==============================================================================
int print(const char *format, ...)
{
   va_list arg;
   int done;

   va_start (arg, format);
   done = vfprintf (stdout, format, arg);
   va_end (arg);

   return done;
}

} // extern

namespace librt {

using namespace contra;
using namespace utils;
using namespace llvm;

//==============================================================================
// Installs the print function
//==============================================================================
const std::string Print::Name = "print";

Function * Print::install(LLVMContext & TheContext, Module & TheModule)
{
  auto PrintType = FunctionType::get(
      Type::getInt32Ty(TheContext),
      llvmType<void*>(TheContext),
      true /* var args */ );

  //auto PrintFun = TheModule.getOrInsertFunction("print", PrintType);
  auto PrintFun = Function::Create(PrintType, Function::ExternalLinkage,
      Print::Name, TheModule);
  return PrintFun;
}


std::unique_ptr<contra::FunctionDef> Print::check()
{
  auto & C = Context::instance();
  std::vector<VariableType> Args;
  Args.emplace_back( C.getStringType() );
  return std::make_unique<BuiltInFunction>(Print::Name, 
      VariableType(C.getVoidType()), Args, true);
}

}
