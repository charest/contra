#include "ast.hpp"
#include "flow.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Visit block
////////////////////////////////////////////////////////////////////////////////

void Flow::visitBlock(const ASTBlock & Block)
{
  for (auto & Stmt : Block)
    runStmtVisitor(*Stmt);
}

  
////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Flow::visit(ValueExprAST<int_t>& e)
{}

//==============================================================================
void Flow::visit(ValueExprAST<real_t>& e)
{}

//==============================================================================
void Flow::visit(ValueExprAST<std::string>& e)
{}

//==============================================================================
void Flow::visit(VarAccessExprAST& e)
{}

//==============================================================================
void Flow::visit(ArrayAccessExprAST& e)
{}

//==============================================================================
void Flow::visit(ArrayExprAST& e)
{}

//==============================================================================
void Flow::visit(CastExprAST& e)
{ runExprVisitor(*e.getFromExpr()); }

//==============================================================================
void Flow::visit(UnaryExprAST& e)
{ runExprVisitor(*e.getOpExpr()); }

//==============================================================================
void Flow::visit(BinaryExprAST& e)
{
  runExprVisitor(*e.getLeftExpr());
  runExprVisitor(*e.getRightExpr());
}

//==============================================================================
void Flow::visit(CallExprAST& e)
{
  for (unsigned i=0; i<e.getNumArgs(); ++i) 
    runExprVisitor(*e.getArgExpr(i));
}

//==============================================================================
void Flow::visit(ForStmtAST& e)
{
  runStmtVisitor(*e.getStartExpr());
  runStmtVisitor(*e.getEndExpr());
  if (e.hasStep())
    runStmtVisitor(*e.getStepExpr());
  visitBlock(e.getBodyExprs());
}

//==============================================================================
void Flow::visit(ForeachStmtAST& e)
{ visit( static_cast<ForStmtAST&>(e) ); }

//==============================================================================
void Flow::visit(IfStmtAST& e)
{
  runStmtVisitor(*e.getCondExpr());
  visitBlock(e.getThenExprs());
  visitBlock(e.getElseExprs());
}

//==============================================================================
void Flow::visit(AssignStmtAST& e)
{
  runExprVisitor(*e.getLeftExpr());
  runExprVisitor(*e.getRightExpr());
}


//==============================================================================
void Flow::visit(VarDeclAST& e)
{
  runExprVisitor(*e.getInitExpr());
  if (e.isArray()) {
    if (e.hasSize()) runExprVisitor(*e.getSizeExpr());
    runExprVisitor(*e.getInitExpr());
  }
}

//==============================================================================
void Flow::visit(PrototypeAST& e)
{}

//==============================================================================
void Flow::visit(FunctionAST& e)
{
  visitBlock(e.getBodyExprs());
  if (e.getReturnExpr()) runStmtVisitor(*e.getReturnExpr());
}

//==============================================================================
void Flow::visit(TaskAST& e)
{ visit(static_cast<FunctionAST&>(e)); }

//==============================================================================
void Flow::visit(IndexTaskAST& e)
{}

}
