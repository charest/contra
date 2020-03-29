#ifndef CONTRA_SYMBOLS_HPP
#define CONTRA_SYMBOLS_HPP

#include "context.hpp"
#include "identifier.hpp"
#include "sourceloc.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace contra {
  
//==============================================================================
// The base symbol type
//==============================================================================
class TypeDef {

  std::string Name_;

public:

  TypeDef(const std::string & Name) : Name_(Name) {}

  virtual ~TypeDef() = default;

  virtual const std::string & getName() const { return Name_; }
  virtual bool isNumber() const { return false; }
};


//==============================================================================
// The builtin symbol type
//==============================================================================
class BuiltInTypeDef : public TypeDef {
  bool IsNumber_ = false;
public:

  BuiltInTypeDef(const std::string & Name, bool IsNumber=false)
    : TypeDef(Name), IsNumber_(IsNumber) {}

  virtual ~BuiltInTypeDef() = default;
  virtual bool isNumber() const override { return IsNumber_; }

};

//==============================================================================
// The builtin symbol type
//==============================================================================
class UserTypeDef : public TypeDef {
  SourceLocation Loc_;
public:

  UserTypeDef(const std::string & Name, const SourceLocation & Loc) : TypeDef(Name),
    Loc_(Loc) {}

  virtual ~UserTypeDef() = default;

  virtual const SourceLocation & getLoc() const { return Loc_; }

};

//==============================================================================
// The variable type
//==============================================================================
class VariableType {

  std::shared_ptr<TypeDef> Type_;
  bool IsArray_ = false;
  bool IsGlobal_ = false;

public:

  VariableType() = default;
  
  explicit VariableType(const VariableType & Type, bool IsArray)
    : Type_(Type.Type_), IsArray_(IsArray)
  {}

  explicit VariableType(std::shared_ptr<TypeDef> Type, bool IsArray = false)
    : Type_(Type), IsArray_(IsArray)
  {}

  //virtual ~VariableType() = default;

  const std::shared_ptr<TypeDef> getBaseType() const { return Type_; }

  bool isArray() const { return IsArray_; }
  void setArray(bool IsArray=true) { IsArray_ = IsArray; }

  bool isGlobal() const { return IsGlobal_; }
  void setGlobal(bool IsGlobal=true) { IsGlobal_ = IsGlobal; }

  bool isNumber() const { return (!IsArray_ && Type_->isNumber()); }

  bool isCastableTo(const VariableType &To) const
  { return (isNumber() && To.isNumber()); }

  bool isAssignableTo(const VariableType &LeftType) const
  {
    if (LeftType == *this) return true;
    if (!LeftType.isArray() && isArray()) return false;
    return isCastableTo(LeftType);
  }

  bool operator==(const VariableType & other)
  { return Type_ == other.Type_ && IsArray_ == other.IsArray_; }
  bool operator!=(const VariableType & other)
  { return Type_ != other.Type_ || IsArray_ != other.IsArray_; }
  
  operator bool() const { return static_cast<bool>(Type_); }

  friend std::ostream &operator<<( std::ostream &out, const VariableType &obj )
  {
    if (obj.IsArray_) out << "[";
     out << obj.Type_->getName();
    if (obj.IsArray_) out << "]";
     return out;
  }
};

using VariableTypeList = std::vector<VariableType>;

//==============================================================================
// The variable symbol
//==============================================================================
class VariableDef : public Identifier, public VariableType {

public:

  VariableDef(const std::string & Name, const SourceLocation & Loc, 
      std::shared_ptr<TypeDef> Type, bool IsArray = false)
    : Identifier(Name, Loc), VariableType(Type, IsArray)
  {}

  VariableDef(const std::string & Name, const SourceLocation & Loc, 
      const VariableType & VarType)
    : Identifier(Name, Loc), VariableType(VarType)
  {}

  VariableType getType() const { return *this; }

  //virtual ~Variable() = default;

};

//==============================================================================
// The function symbol type
//==============================================================================
class FunctionDef {

protected:

  std::string Name_;
  VariableTypeList ArgTypes_;
  VariableType ReturnType_;
  bool IsVarArg_;

public:

  FunctionDef(const std::string & Name, const VariableType & ReturnType,
      const VariableTypeList & ArgTypes, bool IsVarArg = false)
    : Name_(Name), ArgTypes_(ArgTypes), ReturnType_(ReturnType),
      IsVarArg_(IsVarArg)
  {}

  //virtual ~FunctionTypeDef() = default;

  const auto & getName() const { return Name_; }
  const auto & getReturnType() const { return ReturnType_; }
  const auto & getArgTypes() const { return ArgTypes_; }
  const auto & getArgType(int i) const { return ArgTypes_[i]; }
  auto getNumArgs() const { return ArgTypes_.size(); }
  auto isVarArg() const { return IsVarArg_; }
};


//==============================================================================
// The function symbol type
//==============================================================================
class BuiltInFunction : public FunctionDef {

public:

  BuiltInFunction(const std::string & Name, const VariableType & ReturnType, 
      const VariableTypeList & ArgTypes, bool IsVarArg = false)
    : FunctionDef(Name, ReturnType, ArgTypes, IsVarArg)
  {}

};


//==============================================================================
// The function symbol type
//==============================================================================
class UserFunction : public FunctionDef {

  SourceLocation Loc_;

public:

  UserFunction(const std::string & Name, const SourceLocation & Loc,
      const VariableType & ReturnType, const VariableTypeList & ArgTypes,
      bool IsVarArg = false)
    : FunctionDef(Name, ReturnType, ArgTypes, IsVarArg), Loc_(Loc)
  {}
};

} // namespace

#endif // CONTRA_SYMBOLS_HPP
