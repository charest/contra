#include "llvm_includes.hpp"
#include "math.hpp"

#include "config.hpp"
#include "contra/context.hpp"
#include "contra/symbols.hpp"
#include "utils/llvm_utils.hpp"

namespace librt {

using namespace contra;
using namespace utils;
using namespace llvm;

//==============================================================================
// Installs a simple function
//==============================================================================
Function *installDoubleFun(LLVMContext & TheContext, Module & TheModule,
    const std::string & name )
{
  auto FunType = FunctionType::get(
      llvmType<real_t>(TheContext),
      llvmType<real_t>(TheContext) );

  auto Fun = Function::Create(FunType, Function::InternalLinkage,
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
  return std::make_shared<BuiltInFunction>(CSqrt::Name,
      VariableType(Context::F64Type), Args);
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
  return std::make_shared<BuiltInFunction>(CSqrt::Name,
      VariableType(Context::F64Type), Args);
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
  return std::make_shared<BuiltInFunction>(CMax::Name,
      VariableType(Context::F64Type), Args);
}

}
