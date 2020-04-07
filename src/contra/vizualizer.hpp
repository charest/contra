#ifndef CONTRA_VIZUALIZER_HPP
#define CONTRA_VIZUALIZER_HPP

#include "config.hpp"
#include "visiter.hpp"
#include "file_utils.hpp"
#include "string_utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class Vizualizer : public AstVisiter {

  std::ofstream OutputStream_;
  std::ostream * out_ = nullptr;
  int_t ind_ = 0;

public:

  Vizualizer(std::ostream & out = std::cout) : out_(&out)
  {}

  Vizualizer(const std::string & FileName, bool Overwrite = false)
  {
    if (!Overwrite && file_exists(FileName))
      THROW_CONTRA_ERROR("File '" << FileName
          << "' already exists!  Use -f to overwrite.");
    OutputStream_.open(FileName.c_str());
    out_ = &OutputStream_;
  }

  virtual ~Vizualizer()
  {
    stop();
    if (OutputStream_) OutputStream_.close();
  }

  void start() { out() << "digraph {" << std::endl; }
  void stop() { out() << "}" << std::endl; }

  // Codegen function
  template<typename T>
  void runVisitor(T&e)
  {
    e.accept(*this);
  }

private:
   
  void visit(ValueExprAST<int_t>&) override;
  void visit(ValueExprAST<real_t>&) override;
  void visit(ValueExprAST<std::string>&) override;
  void visit(VariableExprAST&) override;
  void visit(ArrayExprAST&) override;
  void visit(CastExprAST&) override;
  void visit(UnaryExprAST&) override;
  void visit(BinaryExprAST&) override;
  void visit(CallExprAST&) override;
  void visit(ForStmtAST&) override;
  void visit(ForeachStmtAST&) override;
  void visit(IfStmtAST&) override;
  void visit(VarDeclAST&) override;
  void visit(ArrayDeclAST&) override;
  void visit(PrototypeAST&) override;
  void visit(FunctionAST&) override;
  void visit(TaskAST&) override;
  
  std::ostream & out() { return *out_; }

  std::string makeLabel(const std::string &, const std::string & = "");

  int_t createLink(int_t, const std::string & = "");

  void labelNode(int_t, const std::string & = "");

  template<typename T>
  void dumpNumericVal(ValueExprAST<T>&);

  void dumpBlock(const ASTBlock &, int_t, const std::string &, bool = false);


};



}

#endif // CONTRA_VIZUALIZER_HPP
