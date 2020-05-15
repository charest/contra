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

  for (unsigned i=0; i<e.getNumPartitions(); ++i) {
    auto PartExpr = dynamic_cast<PartitionStmtAST*>(e.getBodyExpr(i));
    const auto & PartVarName = PartExpr->getVarName();
    PartitionNs.emplace(PartVarName);

    if (!PartExpr->hasBodyExprs()) continue;

    std::string PartTaskName = makeName("partition");
    PartExpr->setTaskName(PartTaskName);

    auto PartDef = PartExpr->getVarDef();
    auto AccessedVars = PartExpr->getAccessedVariables();
    AccessedVars.emplace_back(PartDef);
  
    std::string FieldName = "__"+PartVarName+"_field__";
    auto FieldType = VariableType(PointType_, VariableType::Attrs::Field);
    auto S = std::make_unique<VariableDef>(FieldName, SourceLocation(), FieldType);
    auto res = Context::instance().insertVariable( std::move(S) );
    AccessedVars.emplace_back(res.get());

   
    std::map<std::string, VariableType> VarOverride;
    auto & PartOverrideType = VarOverride[PartVarName];
    PartOverrideType = PartDef->getType();
    PartOverrideType.reset();
    PartOverrideType.setPartition();

    
    // find mine to st partition

    auto PartitionTask = std::make_unique<IndexTaskAST>(
        PartTaskName, 
        std::move(PartExpr->moveBodyExprs()),
        LoopVarName,
        AccessedVars,
        VarOverride);
  
    addFunctionAST(std::move(PartitionTask));
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

  addFunctionAST(std::move(IndexTask));

}

}
