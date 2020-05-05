#include "loops.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

void LoopLifter::postVisit(ForeachStmtAST&e)
{
  std::string Name = makeName();
  e.setLifted();
  e.setName(Name);

  std::set<std::string> PartitionNs;

  for (unsigned i=0; i<e.getNumPartitions(); ++i) {
    auto PartExpr = dynamic_cast<const PartitionStmtAST*>(e.getBodyExpr(i));
    PartitionNs.emplace(PartExpr->getVarName());
  }

  std::vector<bool> VarIsPartition;
  for (auto VarD : e.getAccessedVariables())
    VarIsPartition.emplace_back( PartitionNs.count(VarD->getName()) );

  // lift out the foreach
  auto IndexTask = std::make_unique<IndexTaskAST>(
      Name, 
      std::move(e.moveBodyExprs()),
      e.getVarName(),
      e.getAccessedVariables(),
      VarIsPartition);

  addFunctionAST(std::move(IndexTask));

}

}
