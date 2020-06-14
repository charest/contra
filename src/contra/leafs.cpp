#include "leafs.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void LeafIdentifier::postVisit(CallExprAST& e)
{
  auto FunDef = e.getFunctionDef();
  auto IsTask = FunDef->isTask();
  if (IsTask) CallsTask_ = true;
}

//==============================================================================
void LeafIdentifier::postVisit(AssignStmtAST& e)
{
  for (const auto & Left : e.getLeftExprs()) {
    auto LeftExpr = dynamic_cast<ArrayAccessExprAST*>(Left.get());
    if (LeftExpr) {
      const auto VarDef = LeftExpr->getVariableDef();
      if (VarDef->isField()) {
        auto Index = LeftExpr->getIndexExpr();
        auto IndexExpr = dynamic_cast<ExprAST*>(Index);
        if (IndexExpr && IndexExpr->getType().isRange())
          CallsTask_ = true;
      }
    }
  }
}

//==============================================================================
void LeafIdentifier::postVisit(ForeachStmtAST& e)
{
  if (e.isLifted()) CallsTask_ = true;
}

} // namespace
