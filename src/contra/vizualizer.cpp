#include "ast.hpp"
#include "token.hpp"
#include "vizualizer.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
std::string Vizualizer::makeLabel(const std::string & Type, const std::string & Extra)
{
  std::stringstream ss;
  if (Extra.empty()) {
    ss << "\"" << Type << "\"";
  }
  else {
    auto NewExtra = html(Extra);
    ss << "<" << Type << "<BR />";
    ss << "<FONT POINT-SIZE=\"12\">";
    ss << NewExtra;
    ss << "</FONT>>";
  }
  return ss.str();
}

//==============================================================================
int_t Vizualizer::createLink(int_t From,  const std::string & Label)
{

  out() << "node" << From << " -> node" << ind_+1;
  if (!Label.empty()) out() << " [label=" << Label << "]";
  out() << ";" << std::endl;
  ind_++;
  return ind_;
}

//==============================================================================
void Vizualizer::labelNode(int_t ind, const std::string & Label)
{
  if (!Label.empty())
    out() << "node" << ind << "[label=" << Label <<  "];" << std::endl;

}


//==============================================================================
template<typename T>
void Vizualizer::dumpNumericVal(ValueExprAST<T>& e)
{
  labelNode(ind_, makeLabel(e.getClassName(), Formatter() << e.getVal()));
}

//==============================================================================
void Vizualizer::dumpBlock(const ASTBlock & Block, int_t link_to,
    const std::string & Label, bool ForceExpanded)
{
  auto Num = Block.size();
  if (!Num) return;

  bool IsExpanded = Num>1 || ForceExpanded;

  std::string extra = !IsExpanded ? Label : "";
  
  if (IsExpanded) {
    createLink(link_to);
    labelNode(ind_, Label);
    link_to = ind_;
  }
  for (unsigned i=0; i<Num; ++i) {
    createLink(link_to, extra);
    runVisitor(*Block[i]);
  }
}
  
////////////////////////////////////////////////////////////////////////////////
// Vizitors
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Vizualizer::visit(ValueExprAST<int_t>& e)
{ dumpNumericVal(e); }

//==============================================================================
void Vizualizer::visit(ValueExprAST<real_t>& e)
{ dumpNumericVal(e); }

//==============================================================================
void Vizualizer::visit(ValueExprAST<std::string>& e)
{
  constexpr int MaxLen = 10;
  auto str = e.getVal();
  str.insert(0, "\"");
  if (str.length() > MaxLen+1) {
    str.erase(MaxLen+2, str.length()+1);
    str.append("...");
  }
  str.append("\"");
  labelNode(ind_, makeLabel(e.getClassName(), str));
}

//==============================================================================
void Vizualizer::visit(VarAccessExprAST& e)
{
  Formatter fmt;
  fmt << e.getName();
  if (e.getType()) fmt << " : " << e.getType();

  labelNode(ind_, makeLabel(e.getClassName(), fmt));
}

//==============================================================================
void Vizualizer::visit(ArrayAccessExprAST& e)
{
  Formatter fmt;
  fmt << e.getName() << "[]";
  if (e.getType()) fmt << " : " << e.getType();

  labelNode(ind_, makeLabel(e.getClassName(), fmt));

  createLink(ind_);
  runVisitor(*e.getIndexExpr());
}

//==============================================================================
void Vizualizer::visit(ArrayExprAST& e)
{
  labelNode(ind_, e.getClassName());
}

//==============================================================================
void Vizualizer::visit(CastExprAST& e)
{
  labelNode(ind_, e.getClassName());
}

//==============================================================================
void Vizualizer::visit(UnaryExprAST& e)
{
  labelNode(ind_, e.getClassName());
  createLink(ind_);
  runVisitor(*e.getOpExpr());
}

//==============================================================================
void Vizualizer::visit(BinaryExprAST& e)
{
  auto my_ind = ind_;
  std::string Op = Tokens::getName(e.getOperand());
  labelNode(my_ind, makeLabel(e.getClassName(), Op));
  createLink(my_ind, "Left" );
  runVisitor(*e.getLeftExpr());
  createLink(my_ind, "Right" );
  runVisitor(*e.getRightExpr());
}

//==============================================================================
void Vizualizer::visit(CallExprAST& e)
{
  Formatter fmt;
  fmt << e.getName();
  if (e.getType()) fmt << " : " << e.getType();

  auto my_ind = ind_;
  labelNode( my_ind, makeLabel(e.getClassName(), fmt));
  for (unsigned i=0; i<e.getNumArgs(); ++i) {
    createLink(my_ind, Formatter() << "Arg" << i );    
    runVisitor(*e.getArgExpr(i));
  }
}

//==============================================================================
void Vizualizer::visit(ForStmtAST& e)
{
  auto my_ind = ind_;
  labelNode(my_ind, makeLabel(e.getClassName(), e.getVarName()));
  createLink(my_ind, "Start");
  runVisitor(*e.getStartExpr());
  createLink(my_ind, "End");
  runVisitor(*e.getEndExpr());
  if (e.hasStep()) {
    createLink(my_ind, "Step");
    runVisitor(*e.getStepExpr());
  }
  dumpBlock(e.getBodyExprs(), my_ind, "Body");
}

//==============================================================================
void Vizualizer::visit(ForeachStmtAST& e)
{ visit( static_cast<ForStmtAST&>(e) ); }

//==============================================================================
void Vizualizer::visit(IfStmtAST& e)
{
  auto store_ind = ind_;
  bool force_expanded = (e.getThenExprs().size()>1 || e.getElseExprs().size()>1);
  labelNode(ind_, e.getClassName());

  std::string Label = !force_expanded ? "Cond" : "";
  createLink(ind_, Label);

  if (force_expanded) {
    labelNode(ind_, "Cond");
    createLink(ind_);
  }

  runVisitor(*e.getCondExpr());
  
  dumpBlock(e.getThenExprs(), store_ind, "Then", force_expanded);

  dumpBlock(e.getElseExprs(), store_ind, "Else", force_expanded);
}

//==============================================================================
void Vizualizer::visit(VarDeclAST& e)
{
  Formatter fmt;
  fmt << e.getNames();
  if (e.getType()) fmt << " : " << e.getType();

  labelNode(ind_, makeLabel(e.getClassName(), fmt.str()));
  createLink(ind_, "Init");
  runVisitor(*e.getInitExpr());
}

//==============================================================================
void Vizualizer::visit(ArrayDeclAST& e)
{
  Formatter fmt;
  fmt << e.getNames();
  if (e.getType()) fmt << " : " << e.getType();

  auto my_ind = ind_;
  labelNode(my_ind, makeLabel(e.getClassName(), fmt));
  
  if (e.hasSize()) {
    createLink(my_ind, "Size");
    runVisitor(*e.getSizeExpr());
  }

  createLink(my_ind, "Init");
  e.getInitExpr()->accept(*this);
}

//==============================================================================
void Vizualizer::visit(PrototypeAST& e)
{ labelNode(ind_, e.getClassName()); }

//==============================================================================
void Vizualizer::visit(FunctionAST& e)
{
  auto fun_ind = ind_;
  out() << "subgraph cluster" << fun_ind << " {" << std::endl;
  labelNode(fun_ind, makeLabel(e.getClassName(), e.getName()));

  dumpBlock(e.getBodyExprs(), fun_ind, "Body");

  if (e.getReturnExpr()) {
    auto NumBody = e.getNumBodyExprs();
    std::string Label = NumBody<=1 ? "Return" : "";
    createLink(fun_ind, Label);
    if (NumBody>1) {
      labelNode(ind_, "Return");
      createLink(ind_);
    }
    runVisitor(*e.getReturnExpr());
  }
  out() << "}" << std::endl;

  ind_++;
}

//==============================================================================
void Vizualizer::visit(TaskAST& e)
{ visit(static_cast<FunctionAST&>(e)); }

//==============================================================================
void Vizualizer::visit(IndexTaskAST& e)
{}

}
