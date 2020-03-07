#include "ast.hpp"
#include "vizualizer.hpp"

namespace contra {

//==============================================================================
void Vizualizer::dispatch(ExprAST& e)
{ e.accept(*this); }

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
void Vizualizer::dispatch(BinaryExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"BinaryExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.LHS_);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.RHS_);
}

//==============================================================================
void Vizualizer::dispatch(CallExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"CallExprAST\"];" << std::endl;
  for (int i=0; i<e.Args_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    dispatch(*e.Args_[i]);
  }
}

//==============================================================================
void Vizualizer::dispatch(ForExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"ForExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Start_);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.End_);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Step_);
  for (int i=i; i<e.Body_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    dispatch(*e.Body_[i]);
  }
}

//==============================================================================
void Vizualizer::dispatch(IfExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"IfExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Cond_);
  for (int i=i; i<e.Then_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    dispatch(*e.Then_[i]);
  }
  for (int i=i; i<e.Else_.size(); ++i) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    dispatch(*e.Else_[i]);
  }
}

//==============================================================================
void Vizualizer::dispatch(UnaryExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"UnaryExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Operand_);
}

//==============================================================================
void Vizualizer::dispatch(VarExprAST& e)
{
  out() << "node" << ind_ << "[label=\"VarExprAST\"];" << std::endl;
  out() << "node" << ind_ << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Init_);
}

//==============================================================================
void Vizualizer::dispatch(ArrayVarExprAST& e)
{
  auto my_ind = ind_;
  out() << "node" << my_ind << "[label=\"ArrayVarExprAST\"];" << std::endl;
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Init_);
  out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
  dispatch(*e.Size_);
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
  for ( int i=0; i<e.Body_.size(); ++i ) {
    out() << "node" << my_ind << " -> node" << ++ind_ << ";" << std::endl;
    dispatch(*e.Body_[i]);
  }
  out() << "}" << std::endl;

}

}
