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
void Vizualizer::dumpBlock(ASTBlock & Block, int_t link_to,
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
void Vizualizer::dispatch(ValueExprAST<int_t>& e)
{ dumpNumericVal(e); }

//==============================================================================
void Vizualizer::dispatch(ValueExprAST<real_t>& e)
{ dumpNumericVal(e); }

//==============================================================================
void Vizualizer::dispatch(ValueExprAST<std::string>& e)
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
void Vizualizer::dispatch(VariableExprAST& e)
{
  auto Name = e.getName();
  if (e.isArray()) Name += "[]";

  labelNode(ind_, makeLabel(e.getClassName(), Name));

  if (e.isArray()) {
    createLink(ind_);
    runVisitor(*e.IndexExpr_);
  }
}

//==============================================================================
void Vizualizer::dispatch(ArrayExprAST& e)
{
  labelNode(ind_, e.getClassName());
}

//==============================================================================
void Vizualizer::dispatch(CastExprAST& e)
{
  labelNode(ind_, e.getClassName());
}

//==============================================================================
void Vizualizer::dispatch(UnaryExprAST& e)
{
  labelNode(ind_, e.getClassName());
  createLink(ind_);
  runVisitor(*e.OpExpr_);
}

//==============================================================================
void Vizualizer::dispatch(BinaryExprAST& e)
{
  auto my_ind = ind_;
  std::string Op = Tokens::getName(e.getOperand());
  labelNode(my_ind, makeLabel(e.getClassName(), Op));
  createLink(my_ind, "Left" );
  runVisitor(*e.LeftExpr_);
  createLink(my_ind, "Right" );
  runVisitor(*e.RightExpr_);
}

//==============================================================================
void Vizualizer::dispatch(CallExprAST& e)
{
  auto my_ind = ind_;
  labelNode( my_ind, makeLabel(e.getClassName(), e.getName()));
  for (unsigned i=0; i<e.ArgExprs_.size(); ++i) {
    createLink(my_ind, Formatter() << "Arg" << i );    
    runVisitor(*e.ArgExprs_[i]);
  }
}

//==============================================================================
void Vizualizer::dispatch(ForStmtAST& e)
{
  auto my_ind = ind_;
  labelNode(my_ind, makeLabel(e.getClassName(), e.getVarName()));
  createLink(my_ind, "Start");
  runVisitor(*e.StartExpr_);
  createLink(my_ind, "End");
  runVisitor(*e.EndExpr_);
  if (e.StepExpr_) {
    createLink(my_ind, "Step");
    runVisitor(*e.StepExpr_);
  }
  dumpBlock(e.BodyExprs_, my_ind, "Body");
}

//==============================================================================
void Vizualizer::dispatch(IfStmtAST& e)
{
  auto store_ind = ind_;
  bool force_expanded = (e.ThenExpr_.size()>1 || e.ElseExpr_.size()>1);
  labelNode(ind_, e.getClassName());

  std::string Label = !force_expanded ? "Cond" : "";
  createLink(ind_, Label);

  if (force_expanded) {
    labelNode(ind_, "Cond");
    createLink(ind_);
  }

  runVisitor(*e.CondExpr_);
  
  dumpBlock(e.ThenExpr_, store_ind, "Then", force_expanded);

  dumpBlock(e.ElseExpr_, store_ind, "Else", force_expanded);
}

//==============================================================================
void Vizualizer::dispatch(VarDeclAST& e)
{
  labelNode(ind_, makeLabel(e.getClassName(), Formatter() << e.getNames()));
  createLink(ind_, "Init");
  runVisitor(*e.InitExpr_);
}

//==============================================================================
void Vizualizer::dispatch(ArrayDeclAST& e)
{
  auto my_ind = ind_;
  labelNode(my_ind, makeLabel(e.getClassName(), Formatter() << e.getNames()));
  
  if (e.hasSize()) {
    createLink(my_ind, "Size");
    runVisitor(*e.SizeExpr_);
  }

  createLink(my_ind, "Init");
  e.InitExpr_->accept(*this);
}

//==============================================================================
void Vizualizer::dispatch(PrototypeAST& e)
{ labelNode(ind_, e.getClassName()); }

//==============================================================================
void Vizualizer::dispatch(FunctionAST& e)
{
  auto fun_ind = ind_;
  out() << "subgraph cluster" << fun_ind << " {" << std::endl;
  labelNode(fun_ind, makeLabel(e.getClassName(), e.getName()));

  dumpBlock(e.BodyExprs_, fun_ind, "Body");

  if (e.ReturnExpr_) {
    auto NumBody = e.BodyExprs_.size();
    std::string Label = NumBody<=1 ? "Return" : "";
    createLink(fun_ind, Label);
    if (NumBody>1) {
      labelNode(ind_, "Return");
      createLink(ind_);
    }
    runVisitor(*e.ReturnExpr_);
  }
  out() << "}" << std::endl;

  ind_++;
}

//==============================================================================
void Vizualizer::dispatch(TaskAST& e)
{ dispatch(static_cast<FunctionAST&>(e)); }

}
