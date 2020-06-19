#include "context.hpp"
#include "loops.hpp"
#include "reductions.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////
  
void LoopLifter::postVisit(ForeachStmtAST&e)
{
  const auto & LoopVarName = e.getVarName();

  std::vector<ReductionDef> ReduceVars;

  // determine the reductions
  if (e.hasReduction()) {
    auto NumQual = e.getNumQualifiers();
    for (unsigned i=0; i<NumQual; ++i) {
      auto ReduceExpr = dynamic_cast<ReductionStmtAST*>(e.getBodyExpr(i));
      if (ReduceExpr) {
        auto NumReduceVars = ReduceExpr->getNumVars();
        const auto & OpName = ReduceExpr->getOperatorName();
        auto ReduceOp = SupportedReductions::getType( OpName );
        for (unsigned j=0; j<NumReduceVars; ++j)
          ReduceVars.emplace_back( ReduceExpr->getVarDef(j), ReduceOp );
      }
    }
    e.setReductionVars( ReduceVars );
  }
    

  // lift out the foreach
  std::string LoopTaskName = makeName("loop");
  e.setLifted();
  e.setName(LoopTaskName);

  auto IndexTask = std::make_unique<IndexTaskAST>(
      LoopTaskName, 
      e.moveBodyExprs(),
      LoopVarName,
      e.getAccessedVariables(),
      ReduceVars);

  addFunctionAST(std::move(IndexTask));

}

}
