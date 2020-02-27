#ifndef CONTRA_VARTYPE_HPP
#define CONTRA_VARTYPE_HPP

#include <string>

namespace contra {

//==============================================================================
// Keep track of variable types
//==============================================================================
enum class VarTypes {
  Int,
  Real,
  String,
  Void
};

const VarTypes VarTypesList[] = {
  VarTypes::Int,
  VarTypes::Real,
  VarTypes::String,
  VarTypes::Void
};
  


/// Return the string corresponding to the variable type
std::string getVarTypeName( VarTypes Type );

VarTypes getVarType( const std::string & );
VarTypes getVarType( int tok );

} // namespace

#endif // CONTRA_VARTYPE_HPP
