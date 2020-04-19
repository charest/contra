#ifndef CONTRA_CONTEXT_HPP
#define CONTRA_CONTEXT_HPP

#include "errors.hpp"
#include "symbols.hpp"

#include <deque>
#include <memory>
#include <set>

namespace contra {

class Context {

  using TypeTable = SymbolTable<TypeDef>;
  using VariableTable = SymbolTable<VariableDef>;
  using FunctionTable = SymbolTable<FunctionDef>;

  struct NestedData {

    NestedData* Parent = nullptr;
    std::deque<NestedData*> Children;
    unsigned Level = 0;
    VariableTable Variables;
    std::map<VariableDef*, VariableTable*> AccessedVariables;

    NestedData(const std::string & Name) : Variables(Level, Name) {}

    NestedData(NestedData* Scope, unsigned Lev, const std::string & Name) :
      Parent(Scope), Level(Lev),
      Variables(Lev, Name, &Scope->Variables)
    { Parent->addChild(this); }

  template<typename Op>
  static void decend(NestedData* Start, Op && op)
  {
    if (!Start) return;
    std::forward<Op>(op)(Start);
    for (auto Child : Start->Children) decend(Child, std::forward<Op>(op));
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
  auto getType(const std::string & Name) { return TypeTable_.find(Name); }

  // Variable interface
  auto getVariable(const std::string & Name)
  {
    auto it = CurrentScope_->Variables.find(Name);
    if (it)
      CurrentScope_->AccessedVariables.emplace(it.get(), it.getTable());
    return it;
  }

  auto insertVariable(std::unique_ptr<VariableDef> V)
  { return CurrentScope_->Variables.insert(std::move(V)); }

  std::vector<VariableDef*> getVariablesAccessedFromAbove() const;

  // function interface
  void eraseFunction(const std::string & Name) { return FunctionTable_.erase(Name); }
  auto getFunction(const std::string & Name) { return FunctionTable_.find(Name); }
  auto insertFunction(std::unique_ptr<FunctionDef> F)
  { return FunctionTable_.insert(std::move(F)); }

};

}

#endif // CONTRA_CONTEXT_HPP
