#ifndef CONTRA_SYMBOLS_HPP
#define CONTRA_SYMBOLS_HPP

#include "identifier.hpp"
#include "sourceloc.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class Type;
}

namespace contra {

//==============================================================================
// The attributes class
//==============================================================================
struct Attributes {
  using value_type = std::size_t;

  constexpr Attributes(value_type Val) : Val_(Val) {}
  
  bool operator==(const Attributes & other) const
  { return Val_ == other.Val_; }

  bool operator!=(const Attributes & other) const
  { return Val_ != other.Val_; }

  auto operator | (Attributes rhs) const
  { return Attributes( Val_ | rhs.Val_); }
  
  auto operator & (Attributes rhs) const
  { return Attributes( Val_ & rhs.Val_); }

  auto& operator |= (Attributes rhs) 
  { 
    Val_ |= rhs.Val_;
    return *this;
  }
  
  auto& operator &= (Attributes rhs) 
  { 
    Val_ &= rhs.Val_;
    return *this;
  }

  auto operator~() const
  { return Attributes(~Val_); }
  
  friend std::ostream &operator<<( std::ostream &out, const Attributes &obj )
  {
    out << obj.Val_;
    return out;
  }
  
private:
  value_type Val_ = 0;
};

//==============================================================================
// The base symbol type
//==============================================================================
class TypeDef {

public:

  struct Attr : public Attributes {
    static constexpr Attributes None = 0x00;
    static constexpr Attributes Number = 0x01;
  };

public:

  TypeDef(const std::string & Name, Attributes Attrs=Attr::None)
    : Name_(Name), Attrs_(Attrs) {}

  virtual ~TypeDef() = default;

  virtual const std::string & getName() const { return Name_; }
  virtual bool isNumber() const
  { return ((Attrs_ & Attr::Number) == Attr::Number); }

private:

  TypeDef(const std::string &, bool) {}

  std::string Name_;
  Attributes Attrs_ = Attr::None;
};


//==============================================================================
// The builtin symbol type
//==============================================================================
class BuiltInTypeDef : public TypeDef {
public:

  BuiltInTypeDef(const std::string & Name, Attributes Attrs = Attr::None)
    : TypeDef(Name, Attrs) {}

  virtual ~BuiltInTypeDef() = default;

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

  TypeDef* Type_ = nullptr;
  Attributes Attrs_ = Attr::None;

public:

  struct Attr : public Attributes {
    static constexpr Attributes None   = 0;
    static constexpr Attributes Array  = 1;
    static constexpr Attributes Future = 2;
    static constexpr Attributes Global = 4;
    static constexpr Attributes Range  = 8;
    static constexpr Attributes Field  = 16;
  };

  VariableType() = default;
  
  VariableType(const VariableType & Type)
    : Type_(Type.Type_), Attrs_(Type.Attrs_)
  {}
  
  explicit VariableType(const VariableType & Type, Attributes Attrs)
    : Type_(Type.Type_), Attrs_(Attrs)
  {}

  explicit VariableType(TypeDef* Type, Attributes Attrs = Attr::None)
    : Type_(Type), Attrs_(Attrs)
  {}

  //virtual ~VariableType() = default;

  const TypeDef* getBaseType() const { return Type_; }

  bool isArray() const { return ((Attrs_ & Attr::Array) == Attr::Array); }
  void setArray(bool IsArray=true) {
    if (IsArray) Attrs_ |= Attr::Array;
    else Attrs_ &= ~Attr::Array;
  }

  bool isGlobal() const { return ((Attrs_ & Attr::Global) == Attr::Global); }
  void setGlobal(bool IsGlobal=true) {
    if (IsGlobal) Attrs_ |= Attr::Global;
    else Attrs_ &= ~Attr::Global;
  }
  
  bool isFuture() const { return ((Attrs_ & Attr::Future) == Attr::Future); }
  void setFuture(bool IsFuture=true) {
    if (IsFuture) Attrs_ |= Attr::Future;
    else Attrs_ &= ~Attr::Future;
  }
  
  bool isRange() const { return ((Attrs_ & Attr::Range) == Attr::Range); }
  void setRange(bool IsRange=true) {
    if (IsRange) Attrs_ |= Attr::Range;
    else Attrs_ &= ~Attr::Range;
  }
  
  bool isField() const { return ((Attrs_ & Attr::Field) == Attr::Field); }
  void setField(bool IsField=true) {
    if (IsField) Attrs_ |= Attr::Field;
    else Attrs_ &= ~Attr::Field;
  }

  bool isNumber() const { return (!isArray() && !isRange() && Type_->isNumber()); }

  bool isCastableTo(const VariableType &To) const
  { return (isNumber() && To.isNumber()); }

  bool isAssignableTo(const VariableType &LeftType) const
  {
    if (LeftType == *this) return true;
    if (!LeftType.isArray() && isArray()) return false;
    return isCastableTo(LeftType);
  }

  bool operator==(const VariableType & other)
  { return !(*this == other); }

  bool operator!=(const VariableType & other)
  { return Type_ != other.Type_ || Attrs_ != other.Attrs_; }
  
  operator bool() const { return Type_; }

  friend std::ostream &operator<<( std::ostream &out, const VariableType &obj )
  {
    if (obj.Type_) {
      if (obj.isFuture()) out << "(F)";
      if (obj.isField()) out << "(Fld)";
      if (obj.isArray()) out << "[";
      out << obj.Type_->getName();
      if (obj.isArray()) out << "]";
    }
    else {
      out << "undef";
    }
    return out;
  }

private:

  VariableType(const VariableType &, bool) {}
  VariableType(TypeDef*, bool) {}

};

using VariableTypeList = std::vector<VariableType>;

inline VariableType strip(const VariableType& Ty)
{ return VariableType(Ty, VariableType::Attr::None); }

inline VariableType setArray(const VariableType& Ty, bool IsArray)
{
  VariableType NewTy(Ty);
  NewTy.setArray(IsArray);
  return NewTy;
}

inline VariableType setField(const VariableType& Ty, bool IsField)
{
  VariableType NewTy(Ty);
  NewTy.setField(IsField);
  return NewTy;
}



//==============================================================================
// The variable symbol
//==============================================================================
class VariableDef : public Identifier, public VariableType {

public:

  VariableDef(const std::string & Name, const SourceLocation & Loc, 
      TypeDef* Type, Attributes Attrs = Attr::None)
    : Identifier(Name, Loc), VariableType(Type, Attrs)
  {}

  VariableDef(const std::string & Name, const SourceLocation & Loc, 
      const VariableType & VarType)
    : Identifier(Name, Loc), VariableType(VarType)
  {}

  const VariableType & getType() const { return *this; }
  VariableType& getType() { return *this; }

  //virtual ~Variable() = default;
  
private:

  VariableDef(const std::string &, const SourceLocation &, TypeDef*, bool) {}

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
  Attributes Attrs_;

public:
  
  struct Attr : public Attributes {
    static constexpr Attributes None = 0x00;
    static constexpr Attributes Task  = 0x01;
  };

  FunctionDef(const std::string & Name, const VariableType & ReturnType,
      const VariableTypeList & ArgTypes, bool IsVarArg = false)
    : Name_(Name), ArgTypes_(ArgTypes), ReturnType_(ReturnType),
      IsVarArg_(IsVarArg), Attrs_(Attr::None)
  {}

  //virtual ~FunctionTypeDef() = default;

  const auto & getName() const { return Name_; }
  const auto & getReturnType() const { return ReturnType_; }
  const auto & getArgTypes() const { return ArgTypes_; }
  const auto & getArgType(int i) const { return ArgTypes_[i]; }
  auto getNumArgs() const { return ArgTypes_.size(); }
  auto isVarArg() const { return IsVarArg_; }
  
  bool isTask() const { return ((Attrs_ & Attr::Task) == Attr::Task); }
  void setTask(bool IsTask=true) {
    if (IsTask) Attrs_ |= Attr::Task;
    else Attrs_ &= ~Attr::Task;
  }
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

//==============================================================================
// The symbol table
//==============================================================================
template<typename T>
class SymbolTable {

  std::map<std::string, std::unique_ptr<T>> LookupTable_;
  SymbolTable<T>* Parent_ = nullptr;
  unsigned Level_ = 0;
  std::string Name_;

public:

  class InsertResult {
    T* Pointer_;
    bool IsInserted_;
  public:
    InsertResult(T* Pointer, bool IsInerted)
      : Pointer_(Pointer), IsInserted_(IsInerted) {}
    auto get() const { return Pointer_; }
    auto isInserted() const { return IsInserted_; }
  };
  
  class FindResult {
    T* Pointer_;
    bool IsFound_;
    unsigned Level_;
    SymbolTable* Table_;
  public:
    FindResult(T* Pointer, bool IsFound, SymbolTable* Table)
      : Pointer_(Pointer), IsFound_(IsFound), Table_(Table) {}
    auto get() const { return Pointer_; }
    auto isFound() const { return IsFound_; }
    auto getTable() const { return Table_; }
    operator bool() { return IsFound_; }
  };

  SymbolTable(
      unsigned Level,
      const std::string & Name = "",
      SymbolTable* Parent=nullptr) :
    Parent_(Parent), Level_(Level), Name_(Name)
  {}

  InsertResult insert(std::unique_ptr<T> Symbol) {
    // search first
    const auto & Name = Symbol->getName();
    auto it = find(Name);
    if (it) return {it.get(), false};
    // otherwise insert
    auto res = LookupTable_.emplace(Symbol->getName(), std::move(Symbol));
    return {res.first->second.get(), res.second};
  }

  FindResult find(const std::string & Name) {
    auto it = LookupTable_.find(Name);
    if (it == LookupTable_.end())  {
      if (Parent_) return Parent_->find(Name);
      else return {nullptr, false, nullptr};
    }
    return {it->second.get(), true, this};
  }

  void erase(const std::string & Name) {
    auto it = LookupTable_.find(Name);
    if (it == LookupTable_.end())  {
      if (Parent_) return Parent_->erase(Name);
    }
    else {
      LookupTable_.erase(it);
    }
  }
    
  auto getLevel() const { return Level_; }

};

} // namespace

#endif // CONTRA_SYMBOLS_HPP
