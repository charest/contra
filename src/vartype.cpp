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
/// Return the string corresponding to the variable type
//==============================================================================
std::string getVarTypeName( VarTypes Type ) {
  switch (Type) {
  case VarTypes::Void:
    return "void";
  case VarTypes::Int:
    return getTokName(tok_int);
  case VarTypes::Real:
    return getTokName(tok_real);
  }
  return {};
}

//==============================================================================
// return the LLVM type corresponding with the var type 
//==============================================================================
Type* getLLVMType(VarTypes VarType, LLVMContext & TheContext)
{
  switch (VarType) {
  case VarTypes::Int:
    return Type::getInt64Ty(TheContext);
    break;
  case VarTypes::Real:
    return Type::getDoubleTy(TheContext);
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
    if (getVarTypeName(t) == getTokName(tok))
      return t;
  THROW_CONTRA_ERROR("Unknown variable type '" << getTokName(tok) << "'");
  return VarTypes::Void;
}


} // namespace
