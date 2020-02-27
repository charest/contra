#include "errors.hpp"
#include "token.hpp"
#include "vartype.hpp"

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

VarTypes getVarType( const std::string & str )
{
  for ( const auto & t : VarTypesList )
    if (getVarTypeName(t) == str)
      return t;
  THROW_CONTRA_ERROR("Unknown variable type '" << str << "'");
  return VarTypes::Void;
}


VarTypes getVarType( int tok )
{
  for ( const auto & t : VarTypesList )
    if (getVarTypeName(t) == getTokName(tok))
      return t;
  THROW_CONTRA_ERROR("Unknown variable type '" << getTokName(tok) << "'");
  return VarTypes::Void;
}


} // namespace
