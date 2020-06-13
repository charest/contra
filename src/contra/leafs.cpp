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
void LeafIdentifier::postVisit(ForeachStmtAST& e)
{
  if (e.isLifted()) CallsTask_ = true;
}

//==============================================================================
void LeafIdentifier::postVisit(IndexTaskAST& e)
{
  if (e.hasAutomaticPartitions()) CallsTask_ = true;
}

} // namespace
