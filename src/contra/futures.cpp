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
  auto NumLeft = e.getNumLeftExprs();
  auto NumRight = e.getNumRightExprs();
  
  for (unsigned il=0, ir=0; il<NumLeft; il++) {

    auto RightExpr = e.getRightExpr(ir);
    auto LeftExpr = e.getLeftExpr(il);

    RightExpr->accept(*this);

    auto IsFuture = RightExpr->isFuture();
    if (IsFuture)
      LeftExpr->setFuture();

    auto RightVarAST = dynamic_cast<VarAccessExprAST*>(RightExpr);
    auto LeftVarAST = dynamic_cast<VarAccessExprAST*>(LeftExpr);

    if (LeftVarAST) {
      auto LeftVarDef = LeftVarAST->getVariableDef();
      if (IsFuture) LeftVarDef->getType().setFuture(); 
      if (RightVarAST) addFlow(LeftVarDef, RightVarAST->getVariableDef());
    }
      
    if (NumRight>1) ir++;

  } // for
}

} // namespace
