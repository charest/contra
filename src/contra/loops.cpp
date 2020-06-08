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
    const auto & PartVarName = PartExpr->getVarName();

    if (PartExpr->getVarDef()->getType().isField()) {
      FieldPartitions.emplace_back( PartExpr );
    }
    else {
      PartitionNs.emplace(PartVarName);
    }

  }

  std::map<std::string, VariableType> VarOverride;
  for (auto VarD : e.getAccessedVariables()) {
    const auto & VarN = VarD->getName();
    if (PartitionNs.count(VarN)) {
      auto & OverrideType = VarOverride[VarN];
      OverrideType = VarD->getType();
      OverrideType.reset();
      OverrideType.setPartition();
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
      VarOverride);

  e.setFieldPartitions(FieldPartitions);

  addFunctionAST(std::move(IndexTask));

}

}
