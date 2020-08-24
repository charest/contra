
#include "context.hpp"
#include <algorithm>

namespace contra {

//==============================================================================
Context::Context()
{
  // add builtins
  I64Type_ = insertType(std::make_unique<BuiltInTypeDef>("i64", TypeDef::Attr::Number)).get();
  F64Type_ = insertType(std::make_unique<BuiltInTypeDef>("f64", TypeDef::Attr::Number)).get();
  StrType_ = insertType(std::make_unique<BuiltInTypeDef>("string")).get();
  BoolType_ = insertType(std::make_unique<BuiltInTypeDef>("bool")).get();
  VoidType_ = insertType(std::make_unique<BuiltInTypeDef>("void")).get();

  // scoped data
  NestedData_.emplace_back("Global");
  CurrentScope_ = &NestedData_.back();
}

//==============================================================================
FindResult<TypeDef> Context::getType(const std::string & Name)
{
  auto res = TypeTable_.find(Name);
  if (res) 
    return {res.get()->get(), true};
  else
    return {};
}

//==============================================================================
InsertResult<TypeDef> Context::insertType(std::unique_ptr<TypeDef> V)
{ 
  auto res = TypeTable_.insert(V->getName(), std::move(V));
  return {res.get()->get(), res.isInserted()};
}
  
//==============================================================================
InsertResult<VariableDef> Context::insertVariable(std::unique_ptr<VariableDef> V)
{
  auto res = CurrentScope_->Variables.insert(V->getName(), std::move(V));
  return {res.get()->get(), res.isInserted()};
}

//==============================================================================
FindResult<VariableDef> Context::getVariable(
    const std::string & Name,
    bool Quietly)
{
  VariableTable::FindResultType res;
  NestedData* FoundScope = nullptr;

  NestedData::ascend(
    CurrentScope_,
    [&](auto Scope) { 
      res = Scope->Variables.find(Name);
      if (res) {
        FoundScope = Scope;
        return true;
      }
      return false;
    }
  );
  if (res) {
    if (!Quietly)
      CurrentScope_->AccessedVariables.emplace(res.get()->get(), FoundScope);
    return {res.get()->get(), res.isFound()};
  }
  else {
    return {};
  }
}

//==============================================================================
std::vector<VariableDef*> Context::getVariablesAccessedFromAbove() const
{
  std::vector<VariableDef*> Vars;
  NestedData::decend(
    CurrentScope_,
    [&](auto Scope) { 
      for (const auto & Vpair : Scope->AccessedVariables) {
        if (Vpair.second->Level < CurrentScope_->Level)
          Vars.emplace_back(Vpair.first);
      }
      return false;
    }
  );
  std::sort(Vars.begin(), Vars.end());
  auto last = std::unique(Vars.begin(), Vars.end());
  Vars.erase(last, Vars.end());
  
  std::sort(
      Vars.begin(),
      Vars.end(),
      [](const auto a, const auto b) {
        const auto & astr = a->getName();
        const auto & bstr = b->getName();
        return astr < bstr;
      });

  return Vars;
}
  
//==============================================================================
InsertResult<FunctionDef>
Context::insertFunction(std::unique_ptr<FunctionDef> F)
{ 
  auto & Entry = FunctionTable_[F->getName()];
  // look for existing
  for (const auto & Fs : Entry) {
    if ( isSame(Fs.get(), F.get()) )
      return {Fs.get(), false};
  }
  // otherwise insert
  Entry.emplace_back(std::move(F));
  return {Entry.back().get(), true};
}


}
