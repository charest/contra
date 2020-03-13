#include "ast.hpp"

using namespace llvm;

namespace contra {

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
std::unique_ptr<NodeAST> IfStmtAST::makeNested( 
  std::list< std::pair<SourceLocation, std::unique_ptr<NodeAST>> > & Conds,
  ASTBlockList & Blocks )
{
  auto TopCond = std::move(Conds.front());
  Conds.pop_front();

  auto TopIf = std::make_unique<IfStmtAST>( TopCond.first, std::move(TopCond.second),
      std::move(Blocks.front()) );
  Blocks.pop_front();

  if ( !Blocks.empty() ) {
    if ( Conds.empty() )
      TopIf->ElseExpr_ = std::move(Blocks.back());
    else
      TopIf->ElseExpr_.emplace_back( IfStmtAST::makeNested( Conds, Blocks ) );
  }

  return std::move(TopIf);
}

//------------------------------------------------------------------------------
void IfStmtAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void ForStmtAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void VarDeclAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void ArrayDeclAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void PrototypeAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

//==============================================================================
void FunctionAST::accept(AstDispatcher& dispatcher)
{ dispatcher.dispatch(*this); }

} // namespace
