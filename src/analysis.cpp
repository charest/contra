#include "ast.hpp"
#include "analysis.hpp"

namespace contra {

//==============================================================================
// Get the function
//==============================================================================
std::shared_ptr<FunctionSymbol> Analyzer::getFunction(const std::string & Name) {
  
  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FP = FunctionTable_.find(Name);
  if (FP != FunctionTable_.end()) 
    return FP->second;
  
  // if found it, make sure its not a variable in scope
  return nullptr;
}

//==============================================================================
void Analyzer::dispatch(ExprAST& e)
{ e.accept(*this); }

//==============================================================================
void Analyzer::dispatch(ValueExprAST<int_t>& e)
{}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<real_t>& e)
{}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<std::string>& e)
{}

//==============================================================================
void Analyzer::dispatch(VariableExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(ArrayExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(BinaryExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(CallExprAST& e)
{ }

//==============================================================================
void Analyzer::dispatch(ForExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(IfExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(UnaryExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(VarExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(ArrayVarExprAST& e)
{}

//==============================================================================
void Analyzer::dispatch(PrototypeAST& e)
{
  auto FnName = e.getName();
  auto NumArgs = e.Args_.size();

  std::vector<std::shared_ptr<VariableSymbol>> Args;
  Args.reserve( NumArgs );
  
  for (int i=0; i<NumArgs; ++i) {
    // check type specifier
    const auto & Type = e.ArgTypes_[i];
    auto sit = SymbolTable_.find(Type.Name);
    if ( sit == SymbolTable_.end() )
      THROW_NAME_ERROR("Unknown type specifier '" << Type.Name << "' in prototype"
          " for function '" << FnName << "'.", Type.Loc);
    auto ArgTy = sit->second;
    // check arg name specifier
    const auto & Arg = e.Args_[i];
    auto vit = VariableTable_.find(Arg.Name);
    if (vit != VariableTable_.end() )
      THROW_NAME_ERROR("Variable '" << Arg.Name << "' in prototype"
          " for function '" << FnName << "' previously defined.", Arg.Loc);
    auto S = std::make_shared<VariableSymbol>( Arg.Name, Arg.Loc, ArgTy, e.ArgIsArray_[i]);
    Args.push_back(std::move(S));
  }

  std::shared_ptr<Symbol> RetTy;
 
  if (e.Return_) { 
    auto Ret = *e.Return_;
    auto it = SymbolTable_.find(Ret.Name);
    if ( it == SymbolTable_.end() )
      THROW_NAME_ERROR("Unknown return type specifier '" << Ret.Name << "' in prototype"
          " for function '" << FnName << "'.", Ret.Loc);
    RetTy = it->second;
  }

  auto Sy = std::make_shared<FunctionSymbol>(FnName, e.Loc_, Args, RetTy);
  auto fit = FunctionTable_.emplace( FnName, std::move(Sy) );
  if (!fit.second)
    THROW_NAME_ERROR("Prototype already exists for '" << FnName << "'.",
        e.Loc_);

}

//==============================================================================
void Analyzer::dispatch(FunctionAST& e)
{
  Scope_ = 1;
  auto FnName = e.Proto_->getName();
  auto Loc = e.Proto_->Loc_;

  dispatch(*e.Proto_);
  auto Proto = getFunction(FnName);
  if (!Proto)
    THROW_NAME_ERROR("No valid prototype for '" << FnName << "'.", Loc);

  auto Args = Proto->getArgTypes();
  auto NumArgs = Args.size();

  // If this is an operator, install it.
  //if (P.isBinaryOp())
  //  BinopPrecedence_->operator[](P.getOperatorName()) = P.getBinaryPrecedence();

  // Record the function arguments in the NamedValues map.
  VariableTable_.clear();
  for (unsigned i=0; i<NumArgs; ++i) {
    const auto & Arg = Args[i];
    const auto & Name = Arg->getName();
    auto it = VariableTable_.emplace( Arg->getName(), Arg );
    if (!it.second)
      THROW_NAME_ERROR("Duplicate definition for argument " << i+1
          << ", '" << Name << "' of function '" << FnName << "'", Loc);
  }

  for ( auto & B : e.Body_ ) dispatch(*B);
  if (e.Return_) dispatch(*e.Return_);
  // check return type
}

}
