#include "config.hpp"
#include "llvm_includes.hpp"

#include <math.h>

extern "C" {

//==============================================================================
/// square root function
//==============================================================================
double mysqrt(double x)
{ return sqrt(x); }

//==============================================================================
/// Absolute value function
//==============================================================================
double myabs(double x)
{ return fabs(x); }

} // extern

namespace librt {

using namespace llvm;

//==============================================================================
// Installs a simple function
//==============================================================================
Function *installDoubleFun(LLVMContext & TheContext, Module & TheModule,
    const std::string & name )
{
  auto FunType = FunctionType::get(
      llvmRealType(TheContext),
      llvmRealType(TheContext) );

  auto Fun = Function::Create(FunType, Function::ExternalLinkage,
      name, TheModule);
  return Fun;
}

//==============================================================================
// Installs the c functions
//==============================================================================
Function *installCSqrt(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, "sqrt"); }

Function *installCAbs(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, "fabs"); }

Function *installCMax(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, "fmax"); }

//==============================================================================
// Installs the sqrt function
//==============================================================================
Function *installSqrt(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, "mysqrt"); }


//==============================================================================
// Installs the abs function
//==============================================================================
Function *installAbs(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, "myabs"); }


}
