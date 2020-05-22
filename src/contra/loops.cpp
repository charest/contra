#include "context.hpp"
#include "loops.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////
  
bool LoopLifter::preVisit(PartitionStmtAST& e)
{ 
  if (e.hasBodyExprs()) {
    auto ForeachExpr = dynamic_cast<ForeachStmtAST*>(e.getBodyExpr(0));
    if (ForeachExpr) {

      // lift out the foreach
      std::string TaskName = makeName("partition");
      e.setTaskName(TaskName);
      ForeachExpr->setLifted();
      ForeachExpr->setName(TaskName);
    
      const auto & PartVarName = e.getVarName();
      auto AccessedVars = ForeachExpr->getAccessedVariables();

      auto it = std::find_if(
          AccessedVars.begin(),
          AccessedVars.end(),
          [&](auto Def){ return (Def->getName() == PartVarName); } );
      if (it != AccessedVars.end()) AccessedVars.erase(it);
      
      e.setAccessedVariables(AccessedVars);
    
      auto PartDef = e.getVarDef();
      AccessedVars.emplace_back(PartDef);

      auto & ctx = Context::instance();
      auto PointType = ctx.insertType(std::make_unique<BuiltInTypeDef>("point")).get();
      std::string FieldName = "__"+PartVarName+"_field__";
      auto FieldType = VariableType(PointType, VariableType::Field);
      auto S = std::make_unique<VariableDef>(FieldName, LocationRange(), FieldType);
      auto res = Context::instance().insertVariable( std::move(S) );
      AccessedVars.emplace_back(res.get());


      std::map<std::string, VariableType> VarOverride;
      auto & PartOverrideType = VarOverride[PartVarName];
      PartOverrideType = PartDef->getType();
      PartOverrideType.reset();
      PartOverrideType.setPartition();

      auto IndexTask = std::make_unique<IndexTaskAST>(
          TaskName, 
          std::move(ForeachExpr->moveBodyExprs()),
          ForeachExpr->getVarName(),
          AccessedVars,
          VarOverride);

      addFunctionAST(std::move(IndexTask));
    }
  }
  return true;
}

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

    if (!PartExpr->hasBodyExprs()) continue;

    std::string PartTaskName = makeName("partition");
    PartExpr->setTaskName(PartTaskName);

    auto PartDef = PartExpr->getVarDef();
    auto AccessedVars = PartExpr->getAccessedVariables();
    AccessedVars.emplace_back(PartDef);

    auto & ctx = Context::instance();
    auto PointType = ctx.insertType(std::make_unique<BuiltInTypeDef>("point")).get();

    std::string FieldName = "__"+PartVarName+"_field__";
    auto FieldType = VariableType(PointType, VariableType::Field);
    auto S = std::make_unique<VariableDef>(FieldName, LocationRange(), FieldType);
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

  e.setFieldPartitions(FieldPartitions);

  addFunctionAST(std::move(IndexTask));

}

}
