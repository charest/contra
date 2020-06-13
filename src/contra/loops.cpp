#include "context.hpp"
#include "loops.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////
  
void LoopLifter::postVisit(ForeachStmtAST&e)
{
  std::set<std::string> PartitionNs;
  std::vector<NodeAST*> FieldPartitions;
  const auto & LoopVarName = e.getVarName();

  for (unsigned i=0; i<e.getNumPartitions(); ++i) {
    auto PartExpr = dynamic_cast<PartitionStmtAST*>(e.getBodyExpr(i));
    
    auto NumVars = PartExpr->getNumVars();
    for (unsigned i=0; i<NumVars; ++i) {
      const auto & PartVarName = PartExpr->getVarName(i);
      if (PartExpr->getVarDef(i)->getType().isField()) {
        FieldPartitions.emplace_back( PartExpr );
      }
      else {
        PartitionNs.emplace(PartVarName);
      }
    }
  }

  auto HasAutomaticPartitioning = false;

  std::map<std::string, VariableType> VarOverride;
  for (auto VarD : e.getAccessedVariables()) {
    const auto & VarN = VarD->getName();
    if (PartitionNs.count(VarN)) {
      auto & OverrideType = VarOverride[VarN];
      OverrideType = VarD->getType();
      OverrideType.reset();
      OverrideType.setPartition();
    }
    else if (VarD->getType().isRange()) {
      HasAutomaticPartitioning = true;
    }
  }

  // lift out the foreach
  std::string LoopTaskName = makeName("loop");
  e.setLifted();
  e.setName(LoopTaskName);

  auto IndexTask = std::make_unique<IndexTaskAST>(
      LoopTaskName, 
      std::move(e.moveBodyExprs()),
      LoopVarName,
      e.getAccessedVariables(),
      VarOverride,
      HasAutomaticPartitioning);

  e.setFieldPartitions(FieldPartitions);

  addFunctionAST(std::move(IndexTask));

}

}
