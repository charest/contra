#ifndef CONTRA_VARTYPE_HPP
#define CONTRA_VARTYPE_HPP

#include "config.hpp"

#include <string>

namespace llvm {
  class Type;
  class LLVMContext;
}

namespace contra {

// Keep track of variable types
enum class VarTypes {
  Int,
  Real,
  String,
  Void
};

// A list for iteration
const VarTypes VarTypesList[] = {
  VarTypes::Int,
  VarTypes::Real,
  VarTypes::String,
  VarTypes::Void
};

/// convert types to vartypes
template<typename T>
VarTypes getVarType();

/// Return the string corresponding to the variable type
std::string getVarTypeName( VarTypes Type );

// return the LLVM type corresponding with the var type 
llvm::Type* getLLVMType(VarTypes Type, llvm::LLVMContext & TheContext);

// return the vartype matching the provided string
VarTypes getVarType( const std::string & );

// return the vartype matching the provided token
VarTypes getVarType( int tok );

} // namespace

#endif // CONTRA_VARTYPE_HPP
