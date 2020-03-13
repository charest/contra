#include "llvm_includes.hpp"
#include "math.hpp"

#include "contra/config.hpp"
#include "contra/context.hpp"
#include "contra/symbols.hpp"

namespace librt {

using namespace contra;
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
// Installs the sqrt functions
//==============================================================================

const std::string CSqrt::Name = "sqrt";

Function *CSqrt::install(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, CSqrt::Name); }

std::shared_ptr<contra::FunctionDef> CSqrt::check()
{
  std::vector<VariableType> Args;
  Args.emplace_back( Context::F64Type );
  return std::make_shared<BuiltInFunction>(CSqrt::Name, Args,
      VariableType(Context::F64Type));
}

//==============================================================================
// Installs the abs functions
//==============================================================================
const std::string CAbs::Name = "fabs";

Function *CAbs::install(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, CAbs::Name); }

std::shared_ptr<contra::FunctionDef> CAbs::check()
{
  std::vector<VariableType> Args;
  Args.emplace_back( Context::F64Type );
  return std::make_shared<BuiltInFunction>(CSqrt::Name, Args,
      VariableType(Context::F64Type));
}

//==============================================================================
// Installs the max functions
//==============================================================================
const std::string CMax::Name = "fmax";

Function *CMax::install(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, CMax::Name); }

std::shared_ptr<contra::FunctionDef> CMax::check()
{
  std::vector<VariableType> Args;
  Args.emplace_back( Context::F64Type );
  return std::make_shared<BuiltInFunction>(CMax::Name, Args,
      VariableType(Context::F64Type));
}

}
