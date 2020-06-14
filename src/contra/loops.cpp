#include "context.hpp"
#include "loops.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////
  
void LoopLifter::postVisit(ForeachStmtAST&e)
{
  std::set<std::string> PartitionNs;
  const auto & LoopVarName = e.getVarName();

  // lift out the foreach
  std::string LoopTaskName = makeName("loop");
  e.setLifted();
  e.setName(LoopTaskName);

  auto IndexTask = std::make_unique<IndexTaskAST>(
      LoopTaskName, 
      e.moveBodyExprs(),
      LoopVarName,
      e.getAccessedVariables());

  addFunctionAST(std::move(IndexTask));

}

}
