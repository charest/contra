#include "loop.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

void LoopLifter::postVisit(ForeachStmtAST&e)
{
  std::string Name = makeName();
  e.setLifted();
  e.setName(Name);

  // lift out the foreach
  auto IndexTask = std::make_unique<IndexTaskAST>(Name, 
      std::move(e.moveBodyExprs()), e.getVarName(), e.getAccessedVariables());
      addFunctionAST(std::move(IndexTask));

}

}
