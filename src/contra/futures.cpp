#include "futures.hpp"

namespace contra {

//==============================================================================
FutureIdentifier::VariableEntry *
  FutureIdentifier::findVariable(const std::string & Name)
{
  for ( auto & ST : VariableTable_ ) {
    auto it = ST.find(Name);
    if (it != ST.end()) return &it->second;
  }
  return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
bool FutureIdentifier::preVisit(VarDeclAST&e)
{
  auto NumVars = e.getNumVars();
  for ( unsigned i=0; i<NumVars; ++i)
    addVariable(e.getName(i), VariableEntry{i, &e}); 
  return false;
}
void FutureIdentifier::postVisit(VarDeclAST&e)
{
  if (e.getInitExpr()->isFuture()) e.setFuture();
}

//==============================================================================
void FutureIdentifier::visit(AssignStmtAST&e)
{
  e.getRightExpr()->accept(*this);

  if (e.getRightExpr()->isFuture()) {
    e.getLeftExpr()->setFuture();
  }

  e.getLeftExpr()->accept(*this);
}

//==============================================================================
void FutureIdentifier::postVisit(CallExprAST&e)
{
  if (e.getType().isFuture()) e.setFuture();
}

//==============================================================================
void FutureIdentifier::visit(VarAccessExprAST&e)
{
  if (e.getType().isFuture()) e.setFuture();
  auto it = findVariable(e.getName());
  if (e.isFuture()) {
    it->Node->setFuture( it->Id, true );
  }
}
  
//==============================================================================
bool FutureIdentifier::preVisit(ForStmtAST&e)
{
  pushScope();
  return false;
}
void FutureIdentifier::postVisit(ForStmtAST&e)
{ popScope(); }

//==============================================================================
bool FutureIdentifier::preVisit(ForeachStmtAST&e)
{
  pushScope();
  return false;
}

void FutureIdentifier::postVisit(ForeachStmtAST&e)
{ popScope(); }

//==============================================================================
bool FutureIdentifier::preVisit(IfStmtAST&e)
{
  pushScope();
  return false;
}
void FutureIdentifier::postVisit(IfStmtAST&e)
{ popScope(); }

}
