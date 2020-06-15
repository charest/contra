#ifndef CONTRA_CONTEXT_HPP
#define CONTRA_CONTEXT_HPP

#include "errors.hpp"
#include "lookup.hpp"
#include "symbols.hpp"

#include <deque>
#include <memory>
#include <set>

namespace contra {

class Context {

  using TypeTable = LookupTable< std::unique_ptr<TypeDef> >;
  using VariableTable = LookupTable< std::unique_ptr<VariableDef> >;
  using FunctionTable = LookupTable< std::vector<std::unique_ptr<FunctionDef>> >;

  struct NestedData {

    NestedData* Parent = nullptr;
    std::deque<NestedData*> Children;
    unsigned Level = 0;
    VariableTable Variables;
    std::map<VariableDef*, NestedData*> AccessedVariables;

    NestedData(const std::string & Name) {}

    NestedData(NestedData* Scope, unsigned Lev, const std::string & Name) :
      Parent(Scope), Level(Lev)
    { Parent->addChild(this); }

  template<typename Op>
  static void decend(NestedData* Start, Op && op)
  {
    if (!Start) return;
    if (std::forward<Op>(op)(Start)) return;
    for (auto Child : Start->Children) decend(Child, std::forward<Op>(op));
  }
  
  template<typename Op>
  static void ascend(NestedData* Start, Op && op)
  {
    if (!Start) return;
    if (std::forward<Op>(op)(Start)) return;
    if (Start->Parent) ascend(Start->Parent, std::forward<Op>(op));
  }


  private:
    void addChild(NestedData* Scope) { Children.push_back(Scope); }

  };
  
  // Symbol tables 
  TypeTable TypeTable_;
  FunctionTable FunctionTable_;
  std::deque<NestedData> NestedData_;

  NestedData* CurrentScope_ = nullptr;
  
  // builtin types
  TypeDef* I64Type_ = nullptr;
  TypeDef* F64Type_ = nullptr;
  TypeDef* StrType_ = nullptr;
  TypeDef* BoolType_ = nullptr;
  TypeDef* VoidType_ = nullptr;

public:

  Context();

  // meyers singleton
  static Context& instance() {
    static Context obj;
    return obj;
  }

  // get user defined types
  auto getInt64Type() const { return I64Type_; }
  auto getFloat64Type() const { return F64Type_; }
  auto getStringType() const { return StrType_; }
  auto getBoolType() const { return BoolType_; }
  auto getVoidType() const { return VoidType_; }

  // scope interface
  void createScope(const std::string & Name = "") {
    auto Level = CurrentScope_->Level+1;
    NestedData_.emplace_back(CurrentScope_, Level, Name);
    CurrentScope_ = &NestedData_.back();
  }

  void popScope() {
    if (!CurrentScope_) THROW_CONTRA_ERROR("Something went wrong.  No scope above!");
    CurrentScope_ = CurrentScope_->Parent;
  }

  bool isGlobalScope() const { return CurrentScope_->Level == 0; }

  // type interface
  auto isType(const std::string & Name) { return TypeTable_.has(Name); }
  FindResult<TypeDef> getType(const std::string & Name);
  InsertResult<TypeDef> insertType(std::unique_ptr<TypeDef> V);

  // Variable interface
  FindResult<VariableDef> getVariable(const std::string & Name, bool Quietly=false);
  InsertResult<VariableDef> insertVariable(std::unique_ptr<VariableDef> V);
  std::vector<VariableDef*> getVariablesAccessedFromAbove() const;

  // function interface
  void eraseFunction(const std::string & Name) { return FunctionTable_.erase(Name); }
  auto getFunction(const std::string & Name) { return FunctionTable_.find(Name); }
  InsertResult<FunctionDef> insertFunction(std::unique_ptr<FunctionDef> F);

};

}

#endif // CONTRA_CONTEXT_HPP
