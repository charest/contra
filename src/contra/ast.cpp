#include "ast.hpp"

using namespace llvm;

namespace contra {

//==============================================================================
template<>
void IntegerExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
template<>
void RealExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
template<>
void StringExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void VariableExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void ArrayExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void CastExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void UnaryExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void BinaryExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void CallExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
std::unique_ptr<ExprAST> IfExprAST::makeNested( 
  ExprLocPairList & Conds,
  ExprBlockList & Blocks )
{
  auto TopCond = std::move(Conds.front());
  Conds.pop_front();

  auto TopIf = std::make_unique<IfExprAST>( TopCond.Loc, std::move(TopCond.Expr),
      std::move(Blocks.front()) );
  Blocks.pop_front();

  if ( !Blocks.empty() ) {
    if ( Conds.empty() )
      TopIf->ElseExpr_ = std::move(Blocks.back());
    else
      TopIf->ElseExpr_.emplace_back( IfExprAST::makeNested( Conds, Blocks ) );
  }

  return std::move(TopIf);
}

//------------------------------------------------------------------------------
void IfExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void ForExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void VarDefExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void ArrayDefExprAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void PrototypeAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void FunctionAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

} // namespace
