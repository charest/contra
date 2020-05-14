#include "loops.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

void LoopLifter::postVisit(ForeachStmtAST&e)
{
  std::set<std::string> PartitionNs;
  const auto & LoopVarName = e.getVarName();

  for (unsigned i=0; i<e.getNumPartitions(); ++i) {
    auto PartExpr = dynamic_cast<PartitionStmtAST*>(e.getBodyExpr(i));
    const auto & PartVarName = PartExpr->getVarName();
    PartitionNs.emplace(PartVarName);

    if (!PartExpr->hasBodyExprs()) continue;

    std::string PartTaskName = makeName("partition");
    PartExpr->setTaskName(PartTaskName);

    auto AccessedVars = PartExpr->getAccessedVariables();
    std::vector<bool> VarIsPartition(AccessedVars.size()+1, false);

    AccessedVars.emplace_back(PartExpr->getVarDef());
    VarIsPartition.back() = true;
    
    // find mine to st partition

    auto PartitionTask = std::make_unique<IndexTaskAST>(
        PartTaskName, 
        std::move(PartExpr->moveBodyExprs()),
        LoopVarName,
        AccessedVars,
        VarIsPartition);
  
    addFunctionAST(std::move(PartitionTask));
  }

  std::vector<bool> VarIsPartition;
  for (auto VarD : e.getAccessedVariables())
    VarIsPartition.emplace_back( PartitionNs.count(VarD->getName()) );

  // lift out the foreach
  std::string LoopTaskName = makeName("loop");
  e.setLifted();
  e.setName(LoopTaskName);

  auto IndexTask = std::make_unique<IndexTaskAST>(
      LoopTaskName, 
      std::move(e.moveBodyExprs()),
      LoopVarName,
      e.getAccessedVariables(),
      VarIsPartition);

  addFunctionAST(std::move(IndexTask));

}

}
