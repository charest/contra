#include "futures.hpp"

namespace contra {

void FutureIdentifier::runVisitor(FunctionAST&e)
{
  VariableTable_.clear();
  e.accept(*this);

  for ( auto & VarPair : VariableTable_ ) {
    auto VarDef = VarPair.first;
    for (auto OtherVar : VarPair.second) {
      if (OtherVar->getType().isFuture()) {
        VarDef->getType().setFuture();
        break;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void FutureIdentifier::postVisit(CallExprAST& e)
{
  auto FunDef = e.getFunctionDef();
  auto IsTask = FunDef->isTask();

  auto NumArgs = e.getNumArgs();
  for (unsigned i=0; i<NumArgs; ++i) {
    auto ArgExpr = e.getArgExpr(i);
    if (!IsTask) ArgExpr->setFuture(false);
  }

  e.setFuture(IsTask);
}

//==============================================================================
void FutureIdentifier::visit(AssignStmtAST& e)
{
  e.getRightExpr()->accept(*this);

  auto IsFuture = e.getRightExpr()->isFuture();
  if (IsFuture)
    e.getLeftExpr()->setFuture();

  auto RightVarAST = dynamic_cast<VarAccessExprAST*>(e.getRightExpr());
  auto LeftVarAST = dynamic_cast<VarAccessExprAST*>(e.getLeftExpr());

  if (LeftVarAST) {
    auto LeftVarDef = LeftVarAST->getVariableDef();
    if (IsFuture) LeftVarDef->getType().setFuture(); 
    if (RightVarAST) addFlow(LeftVarDef, RightVarAST->getVariableDef());
  }
}

//==============================================================================
void FutureIdentifier::visit(VarDeclAST& e)
{
  e.getInitExpr()->accept(*this);
  auto AreFutures = e.getInitExpr()->isFuture();
  
  auto InitVarAST = dynamic_cast<VarAccessExprAST*>(e.getInitExpr());

  auto NumVars = e.getNumVars();
  for (unsigned i=0; i<NumVars; ++i) {
    if (AreFutures) e.getVarType(i).setFuture();
    if (InitVarAST) addFlow(e.getVariableDef(i), InitVarAST->getVariableDef());
  }
}

//==============================================================================
void FutureIdentifier::visit(FieldDeclAST& e)
{ visit( static_cast<VarDeclAST&>(e) ); }


}
