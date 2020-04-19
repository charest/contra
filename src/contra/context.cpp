
#include "context.hpp"

namespace contra {

//==============================================================================
Context::Context() : TypeTable_(0,"Global"), FunctionTable_(0, "Global")
{
  // add builtins
  I64Type_ =
    TypeTable_.insert(std::make_unique<BuiltInTypeDef>("i64", TypeDef::Attr::Number)).get();
  F64Type_ =
    TypeTable_.insert(std::make_unique<BuiltInTypeDef>("f64", TypeDef::Attr::Number)).get();
  StrType_ =
    TypeTable_.insert(std::make_unique<BuiltInTypeDef>("string")).get();
  BoolType_ =
    TypeTable_.insert(std::make_unique<BuiltInTypeDef>("bool")).get();
  VoidType_ =
    TypeTable_.insert(std::make_unique<BuiltInTypeDef>("void")).get();

  // scoped data
  NestedData_.emplace_back("Global");
  CurrentScope_ = &NestedData_.back();
}

//==============================================================================
std::vector<VariableDef*> Context::getVariablesAccessedFromAbove() const
{
  std::vector<VariableDef*> Vars;
  NestedData::decend(
    CurrentScope_,
    [&](auto Scope) { 
      for (const auto & Vpair : Scope->AccessedVariables) {
        if (Vpair.second->getLevel() < CurrentScope_->Level)
          Vars.emplace_back(Vpair.first);
      }
    }
  );
  return Vars;
}


}
