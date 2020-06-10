#include "ast.hpp"

namespace contra {

//==============================================================================
void ExprListAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
template<>
std::string ValueExprAST::getVal() const
{ return Val_; }

template<>
int_t ValueExprAST::getVal() const
{ return std::atoi(Val_.c_str()); }

template<>
real_t ValueExprAST::getVal() const
{ return std::atof(Val_.c_str()); }

//==============================================================================
void VarAccessExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void ArrayAccessExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void ArrayExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void RangeExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void FieldDeclExprAST::accept(AstVisiter& visiter)
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
  IfStmtAST::ConditionList & Conds,
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
void AssignStmtAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void PartitionStmtAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
//void VarDeclAST::accept(AstVisiter& visiter)
//{ visiter.visit(*this); }

//==============================================================================
//void FieldDeclAST::accept(AstVisiter& visiter)
//{ visiter.visit(*this); }

//==============================================================================
void PrototypeAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void FunctionAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void TaskAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void IndexTaskAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

//==============================================================================
void LambdaExprAST::accept(AstVisiter& visiter)
{ visiter.visit(*this); }

} // namespace
