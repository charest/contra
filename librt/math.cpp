#include "config.hpp"
#include "llvm_includes.hpp"
#include "math.hpp"

#include "src/context.hpp"
#include "src/symbols.hpp"

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
  Args.emplace_back( Context::F64Symbol );
  return std::make_shared<BuiltInFunction>(CSqrt::Name, Args, Context::F64Symbol);
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
  Args.emplace_back( Context::F64Symbol );
  return std::make_shared<BuiltInFunction>(CSqrt::Name, Args, Context::F64Symbol);
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
  Args.emplace_back( Context::F64Symbol );
  return std::make_shared<BuiltInFunction>(CMax::Name, Args, Context::F64Symbol);
}

}
