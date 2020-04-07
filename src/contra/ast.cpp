#include "ast.hpp"

namespace contra {

//==============================================================================
template<>
std::string IntegerExprAST::getClassName() const
{ return "IntegerExprAST"; }

template<>
std::string RealExprAST::getClassName() const
{ return "RealExprAST"; }

template<>
std::string StringExprAST::getClassName() const
{ return "StringExprAST"; }

//==============================================================================
void VariableExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void ArrayExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void CastExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void UnaryExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void BinaryExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void CallExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

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
void IfStmtAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void ForStmtAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void ForeachStmtAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void VarDeclAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void ArrayDeclAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void PrototypeAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void FunctionAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void TaskAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

} // namespace
