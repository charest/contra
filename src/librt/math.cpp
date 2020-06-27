#include "llvm_includes.hpp"
#include "math.hpp"

#include "config.hpp"
#include "contra/context.hpp"
#include "contra/symbols.hpp"
#include "utils/llvm_utils.hpp"

extern "C" {
  
//==============================================================================
/// Integer max/min
//==============================================================================
int_t imax(int_t a, int_t b) 
{ return a > b ? a : b; }

int_t imin(int_t a, int_t b) 
{ return a < b ? a : b; }

} // extern

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

std::unique_ptr<contra::FunctionDef> CSqrt::check()
{
  auto & C = Context::instance();
  std::vector<VariableType> Args;
  Args.emplace_back( C.getFloat64Type() );
  return std::make_unique<BuiltInFunction>(CSqrt::Name,
      VariableType(C.getFloat64Type()), Args);
}

//==============================================================================
// Installs the abs functions
//==============================================================================
const std::string CAbs::Name = "fabs";

Function *CAbs::install(LLVMContext & TheContext, Module & TheModule)
{ return installDoubleFun(TheContext, TheModule, CAbs::Name); }

std::unique_ptr<contra::FunctionDef> CAbs::check()
{
  auto & C = Context::instance();
  std::vector<VariableType> Args;
  Args.emplace_back( C.getFloat64Type() );
  return std::make_unique<BuiltInFunction>(CSqrt::Name,
      VariableType(C.getFloat64Type()), Args);
}

//==============================================================================
// Installs the max functions
//==============================================================================
const std::string CMax::Name = "fmax";

Function *CMax::install(LLVMContext & TheContext, Module & TheModule)
{
  auto RealType = llvmType<real_t>(TheContext);
  std::vector<Type*> Args = {RealType, RealType};
  auto FunType = FunctionType::get( RealType, Args, false );

  auto Fun = Function::Create(FunType, Function::InternalLinkage,
      Name, TheModule);
  return Fun;
}

std::unique_ptr<contra::FunctionDef> CMax::check()
{
  auto & C = Context::instance();
  auto RealType = VariableType(C.getFloat64Type());
  std::vector<VariableType> Args = {RealType, RealType};
  return std::make_unique<BuiltInFunction>(CMax::Name, RealType, Args);
}

//==============================================================================
// Installs the min functions
//==============================================================================
const std::string CMin::Name = "fmin";

Function *CMin::install(LLVMContext & TheContext, Module & TheModule)
{
  auto RealType = llvmType<real_t>(TheContext);
  std::vector<Type*> Args = {RealType, RealType};
  auto FunType = FunctionType::get( RealType, Args, false );

  auto Fun = Function::Create(FunType, Function::InternalLinkage,
      Name, TheModule);
  return Fun;
}

std::unique_ptr<contra::FunctionDef> CMin::check()
{
  auto & C = Context::instance();
  auto RealType = VariableType(C.getFloat64Type());
  std::vector<VariableType> Args = {RealType, RealType};
  return std::make_unique<BuiltInFunction>(CMin::Name, RealType, Args);
}

}
