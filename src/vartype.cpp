#include "config.hpp"
#include "errors.hpp"
#include "token.hpp"
#include "vartype.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"

using llvm::Type;
using llvm::LLVMContext;

namespace contra {

//==============================================================================
/// Specializations to convert c++ types to vartypes
//==============================================================================
template<>
VarTypes getVarType<int_t>() { return VarTypes::Int; }

template<>
VarTypes getVarType<real_t>() { return VarTypes::Real; }

template<>
VarTypes getVarType<std::string>() { return VarTypes::String; }

template<>
VarTypes getVarType<void>() { return VarTypes::Void; }

//==============================================================================
/// Return the string corresponding to the variable type
//==============================================================================
std::string getVarTypeName( VarTypes Type ) {
#if 0
  switch (Type) {
  case VarTypes::Void:
    return "void";
  case VarTypes::Int:
    return Tokens::getName(tok_int);
  case VarTypes::Real:
    return Tokens::getName(tok_real);
  }
#endif
  return {};
}

//==============================================================================
// return the LLVM type corresponding with the var type 
//==============================================================================
Type* getLLVMType(VarTypes VarType, LLVMContext & TheContext)
{
  switch (VarType) {
  case VarTypes::Int:
    return llvmIntegerType(TheContext);
    break;
  case VarTypes::Real:
    return llvmRealType(TheContext);
    break;
  default:
    THROW_CONTRA_ERROR( "Unknown argument type of '" << getVarTypeName(VarType) << "'" );
  case VarTypes::Void:
    return Type::getVoidTy(TheContext);
  }
}

//==============================================================================
// return the vartype matching the provided string
//==============================================================================
VarTypes getVarType( const std::string & str )
{
  for ( const auto & t : VarTypesList )
    if (getVarTypeName(t) == str)
      return t;
  THROW_CONTRA_ERROR("Unknown variable type '" << str << "'");
  return VarTypes::Void;
}


//==============================================================================
// return the vartype matching the provided token
//==============================================================================
VarTypes getVarType( int tok )
{
  for ( const auto & t : VarTypesList )
    if (getVarTypeName(t) == Tokens::getName(tok))
      return t;
  THROW_CONTRA_ERROR("Unknown variable type '" << Tokens::getName(tok) << "'");
  return VarTypes::Void;
}


} // namespace
