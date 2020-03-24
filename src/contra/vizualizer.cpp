#include "ast.hpp"
#include "vizualizer.hpp"

namespace contra {


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
template<typename T>
void Vizualizer::dumpNumericVal(ValueExprAST<T>& e)
{
  std::stringstream ss;
  ss << e.getVal();
  out() << "node" << ind_ << "[label=" << makeLabel(e.getClassName(), ss.str())
    <<  "];" << std::endl;
}

//==============================================================================
void Vizualizer::dumpBlock(ASTBlock & Block, int_t link_to,
    const std::string & Label, bool ForceExpanded)
{
  auto Num = Block.size();
  if (!Num) return;

  bool IsExpanded = Num>1 || ForceExpanded;

  std::string extra = !IsExpanded ? " [label="+Label+"]" : "";
  
  if (IsExpanded) {
    out() << "node" << link_to << " -> node" << ++ind_ << ";" << std::endl; 
    out() << "node" << ind_ << "[label=" << Label << "];" << std::endl;
    link_to = ind_;
  }
  for (unsigned i=0; i<Num; ++i) {
    out() << "node" << link_to << " -> node" << ++ind_ << extra << ";" << std::endl;
    runVisitor(*Block[i]);
  }
}
  
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
  out() << "node" << ind_ << "[label=" << makeLabel(e.getClassName(), str)
    <<  "];" << std::endl;
}

//==============================================================================
void Vizualizer::dispatch(VariableExprAST& e)
{
  auto Name = e.getName();
  if (e.isArray()) Name += "[]";

  out() << "node" << ind_ << "[label=" << makeLabel(e.getClassName(), Name)
    << "];" << std::endl;

  if (e.isArray()) {
    out() << "node" << ind_ << " -> node" << ++ind_ << ";" << std::endl;
    runVisitor(*e.IndexExpr_);
  }
}

//==============================================================================
void Vizualizer::dispatch(ArrayExprAST& e)
{
  out() << "node" << ind_ << "[label=\"ArrayExprAST\"];" << std::endl;
}

//==============================================================================
void Vizualizer::dispatch(CastExprAST& e)
{
  out() << "node" << ind_ << "[label=\"CastExprAST\"];" << std::endl;
}

//==============================================================================
void Vizualizer::dispatch(UnaryExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"UnaryExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.OpExpr_->accept(*this);
}

//==============================================================================
void Vizualizer::dispatch(BinaryExprAST& e)
{
  auto my_ind = ind_;
  std::string Op( 1, e.getOperand() );
  out() << "node" << my_ind << "[label=" << makeLabel(e.getClassName(), Op)
    << "];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << " [label=Left];" << std::endl;
  runVisitor(*e.LeftExpr_);
  out() << "node" << my_ind << " -> node" << ++ind_ << " [label=Right];" << std::endl;
  runVisitor(*e.RightExpr_);
}

//==============================================================================
void Vizualizer::dispatch(CallExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=" << makeLabel(e.getClassName(), e.getName()) 
    << "];" << std::endl;
  for (unsigned i=0; i<e.ArgExprs_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << " [label=Arg" << i
      << "];" << std::endl;
    runVisitor(*e.ArgExprs_[i]);
  }
}

//==============================================================================
void Vizualizer::dispatch(ForStmtAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=" << makeLabel(e.getClassName(),
      e.getVarName()) << "];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << " [label=Start];" << std::endl;
  runVisitor(*e.StartExpr_);
  out() << "node" << my_ind << " -> node" << ++ind_ << " [label=End];" << std::endl;
  runVisitor(*e.EndExpr_);
  if (e.StepExpr_) {
    out() << "node" << my_ind << " -> node" << ++ind_ << " [label=Step];" << std::endl;
    runVisitor(*e.StepExpr_);
  }
  dumpBlock(e.BodyExprs_, my_ind, "Body");
}

//==============================================================================
void Vizualizer::dispatch(IfStmtAST& e)
{
  auto store_ind = ind_;
  bool force_expanded = (e.ThenExpr_.size()>1 || e.ElseExpr_.size()>1);
  out() << "node" << ind_ << "[label=" << e.getClassName() << "];" << std::endl;
  out() << "node" << ind_ << " -> node" << ++ind_;
  if (force_expanded) {
    out() << ";" << std::endl;
    out() << "node" << ind_ << "[label=Cond];" << std::endl;
    out() << "node" << ind_ << " -> node" << ++ind_ << ";" << std::endl;
  }
  else {
    out() << " [label=Cond];" << std::endl;
  }

  runVisitor(*e.CondExpr_);
  
  dumpBlock(e.ThenExpr_, store_ind, "Then", force_expanded);

  dumpBlock(e.ElseExpr_, store_ind, "Else", force_expanded);
}

//==============================================================================
void Vizualizer::dispatch(VarDeclAST& e)
{
  Formatter fmt;
  fmt << e.getNames();
  out() << "node" << ind_ << "[label=" << makeLabel(e.getClassName(), fmt.str())
    << "];" << std::endl;
  out() << "node" << ind_ << " -> node" << ++ind_ << " [label=Init];" << std::endl;
  runVisitor(*e.InitExpr_);
}

//==============================================================================
void Vizualizer::dispatch(ArrayDeclAST& e)
{
  Formatter fmt;
  fmt << e.getNames();
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=" << makeLabel(e.getClassName(), fmt.str()) 
    << "];" << std::endl;
  
  if (e.hasSize()) {
    out() << "node" << my_ind << " -> node" << ++ind_
      << " [label=Size];" << std::endl;
    runVisitor(*e.SizeExpr_);
  }

  out() << "node" << my_ind << " -> node" << ++ind_ << " [label=Init];" << std::endl;
  e.InitExpr_->accept(*this);
}

//==============================================================================
void Vizualizer::dispatch(PrototypeAST& e)
{ std::cout << "PrototypeAST" << std::endl; }

//==============================================================================
void Vizualizer::dispatch(FunctionAST& e)
{
  auto fun_ind = ++ind_;
  out() << "subgraph cluster" << fun_ind << " {" << std::endl;
  out() << "node" << fun_ind << "[label=" <<
    makeLabel(e.getClassName(), e.getName()) << "];" << std::endl;

  dumpBlock(e.BodyExprs_, fun_ind, "Body");

  if (e.ReturnExpr_) {
    out() << "node" << fun_ind << " -> node" << ++ind_;
    auto NumBody = e.BodyExprs_.size();
    if (NumBody>1) {
      out() << ";" << std::endl;
      out() << "node" << ind_ << "[label=Return];" << std::endl;
      out() << "node" << ind_ << " -> node" << ++ind_ << ";" << std::endl;
    }
    else {
      out() << " [label=Return];" << std::endl;
    }
    runVisitor(*e.ReturnExpr_);
  }
  out() << "}" << std::endl;
}

}
