#include "ast.hpp"
#include "vizualizer.hpp"

namespace contra {

//==============================================================================
void Vizualizer::dispatch(ValueExprAST<int_t>& e)
{
  out() << "node" << ind_ << "[label=\"IntegerExprAST\"];" << std::endl;
}

//==============================================================================
void Vizualizer::dispatch(ValueExprAST<real_t>& e)
{
  out() << "node" << ind_ << "[label=\"RealExprAST\"];" << std::endl;
}

//==============================================================================
void Vizualizer::dispatch(ValueExprAST<std::string>& e)
{
  out() << "node" << ind_ << "[label=\"StringExprAST\"];" << std::endl;
}

//==============================================================================
void Vizualizer::dispatch(VariableExprAST& e)
{
  out() << "node" << ind_ << "[label=\"VariableExprAST\"];" << std::endl;
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
  out() << "node" << my_ind << "[label=\"BinaryExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.LeftExpr_->accept(*this);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.RightExpr_->accept(*this);
}

//==============================================================================
void Vizualizer::dispatch(CallExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"CallExprAST\"];" << std::endl;
  for (unsigned i=0; i<e.ArgExprs_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    e.ArgExprs_[i]->accept(*this);
  }
}

//==============================================================================
void Vizualizer::dispatch(ForStmtAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"ForExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.StartExpr_->accept(*this);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.EndExpr_->accept(*this);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.StepExpr_->accept(*this);
  for (unsigned i=0; i<e.BodyExprs_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    e.BodyExprs_[i]->accept(*this);
  }
}

//==============================================================================
void Vizualizer::dispatch(IfStmtAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"IfExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.CondExpr_->accept(*this);
  for (unsigned i=0; i<e.ThenExpr_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    e.ThenExpr_[i]->accept(*this);
  }
  for (unsigned i=0; i<e.ElseExpr_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    e.ElseExpr_[i]->accept(*this);
  }
}

//==============================================================================
void Vizualizer::dispatch(VarDeclAST& e)
{
  out() << "node" << ind_ << "[label=\"VarDefExprAST\"];" << std::endl;
  out() << "node" << ind_ << " -> node" << ind_+1 << ";" << std::endl;
  ind_++;
  e.InitExpr_->accept(*this);
}

//==============================================================================
void Vizualizer::dispatch(ArrayDeclAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"ArrayDefExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.InitExpr_->accept(*this);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  e.SizeExpr_->accept(*this);
}

//==============================================================================
void Vizualizer::dispatch(PrototypeAST& e)
{ std::cout << "PrototypeAST" << std::endl; }

//==============================================================================
void Vizualizer::dispatch(FunctionAST& e)
{
  auto my_ind = ind_;
  out() << "digraph {" << std::endl;
  out() << "node" << my_ind << "[label=\"FunctionAST\"];" << std::endl;
  for ( unsigned i=0; i<e.BodyExprs_.size(); ++i ) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    e.BodyExprs_[i]->accept(*this);
  }
  out() << "}" << std::endl;
}

}
