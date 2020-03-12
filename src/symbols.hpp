#ifndef CONTRA_SYMBOLS_HPP
#define CONTRA_SYMBOLS_HPP

#include "identifier.hpp"
#include "sourceloc.hpp"
#include "vartype.hpp"

#include <map>
#include <string>

namespace contra {
  
//==============================================================================
// The base symbol type
//==============================================================================
class Symbol {

  std::string Name_;

public:

  Symbol(const std::string & Name) : Name_(Name) {}

  virtual ~Symbol() = default;

  virtual const std::string & getName() const { return Name_; }
};


//==============================================================================
// The builtin symbol type
//==============================================================================
class BuiltInSymbol : public Symbol {
public:

  BuiltInSymbol(const std::string & Name) : Symbol(Name) {}

  virtual ~BuiltInSymbol() = default;

};

//==============================================================================
// The builtin symbol type
//==============================================================================
class UserSymbol : public Symbol {
  SourceLocation Loc_;
public:

  UserSymbol(const std::string & Name, SourceLocation Loc) : Symbol(Name),
    Loc_(Loc) {}

  virtual ~UserSymbol() = default;

  virtual SourceLocation getLoc() const { return Loc_; }

};

//==============================================================================
// The variable type
//==============================================================================
class VariableType {

  std::shared_ptr<Symbol> Type_;
  bool IsArray_ = false;

public:

  VariableType() = default;

  VariableType(std::shared_ptr<Symbol> Type, bool IsArray = false)
    : Type_(Type), IsArray_(IsArray)
  {}

  //virtual ~VariableType() = default;

  const std::shared_ptr<Symbol> getSymbol() const { return Type_; }
  bool isArray() const { return IsArray_; }
  void setArray(bool IsArray=true) { IsArray_ = IsArray; }

  bool operator==(const VariableType & other)
  { return Type_ == other.Type_ && IsArray_ == other.IsArray_; }
  bool operator!=(const VariableType & other)
  { return Type_ != other.Type_ || IsArray_ != other.IsArray_; }
};

using VariableTypeList = std::vector<VariableType>;

//==============================================================================
// The variable symbol
//==============================================================================
class VariableDef : public Identifier, public VariableType {

public:

  VariableDef(const std::string & Name, SourceLocation Loc, 
      std::shared_ptr<Symbol> Type, bool IsArray = false)
    : VariableType(Type, IsArray), Identifier(Name, Loc)
  {}

  VariableDef(const std::string & Name, SourceLocation Loc, 
      const VariableType & VarType)
    : VariableType(VarType), Identifier(Name, Loc)
  {}

  //virtual ~Variable() = default;

};

//==============================================================================
// The function symbol type
//==============================================================================
class FunctionDef{

public:


protected:

  std::string Name_;
  VariableTypeList ArgTypes_;
  VariableType ReturnType_;

public:

  FunctionDef(const std::string & Name, const VariableTypeList & ArgTypes)
    : Name_(Name), ArgTypes_(ArgTypes), ReturnType_(Context::VoidSymbol)
  {}

  FunctionDef(const std::string & Name, const VariableTypeList & ArgTypes,
      const VariableType & ReturnType)
    : Name_(Name), ArgTypes_(ArgTypes), ReturnType_(ReturnType)
  {}

  //virtual ~FunctionSymbol() = default;

  const auto & getName() const { return Name_; }
  const auto & getReturnType() const { return ReturnType_; }
  const auto & getArgTypes() const { return ArgTypes_; }
  auto getNumArgs() const { return ArgTypes_.size(); }
};


//==============================================================================
// The function symbol type
//==============================================================================
class BuiltInFunction : public FunctionDef {

public:

  BuiltInFunction(const std::string & Name, const VariableTypeList & ArgTypes)
    : FunctionDef(Name, ArgTypes)
  {}

  BuiltInFunction(const std::string & Name, const VariableTypeList & ArgTypes,
      const VariableType & ReturnType) : FunctionDef(Name, ArgTypes, ReturnType)
  {}

};


//==============================================================================
// The function symbol type
//==============================================================================
class UserFunction : public FunctionDef {

  SourceLocation Loc_;

public:

  UserFunction(const std::string & Name, SourceLocation Loc,
      const VariableTypeList & ArgTypes)
    : FunctionDef(Name, ArgTypes), Loc_(Loc)
  {}

  UserFunction(const std::string & Name, SourceLocation Loc,
      const VariableTypeList & ArgTypes, const VariableType & ReturnType)
    : FunctionDef(Name, ArgTypes, ReturnType), Loc_(Loc)
  {}

  //virtual ~FunctionSymbol() = default;
};

} // namespace

#endif // CONTRA_SYMBOLS_HPP
