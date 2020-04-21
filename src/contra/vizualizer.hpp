#ifndef CONTRA_VIZUALIZER_HPP
#define CONTRA_VIZUALIZER_HPP

#include "config.hpp"
#include "recursive.hpp"
#include "file_utils.hpp"
#include "string_utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class Vizualizer : public RecursiveAstVisiter {

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
  void runVisitor(FunctionAST & e)
  { e.accept(*this); }
  
  void runVisitor(NodeAST & e)
  { e.accept(*this); }

private:
   
  void visit(ValueExprAST&) override;
  void visit(VarAccessExprAST&) override;
  void visit(ArrayAccessExprAST&) override;
  void visit(ArrayExprAST&) override;
  void visit(CastExprAST&) override;
  void visit(UnaryExprAST&) override;
  void visit(BinaryExprAST&) override;
  void visit(CallExprAST&) override;
  void visit(ForStmtAST&) override;
  void visit(ForeachStmtAST&) override;
  void visit(IfStmtAST&) override;
  void visit(AssignStmtAST&) override;
  void visit(VarDeclAST&) override;
  void visit(FieldDeclAST&) override;
  void visit(PrototypeAST&) override;
  void visit(FunctionAST&) override;
  void visit(TaskAST&) override;
  void visit(IndexTaskAST&) override;
  
  std::ostream & out() { return *out_; }

  std::string makeLabel(const std::string &, const std::string & = "");

  int_t createLink(int_t, const std::string & = "");

  void labelNode(int_t, const std::string & = "");

  void dumpBlock(const ASTBlock &, int_t, const std::string &, bool = false);

};



}

#endif // CONTRA_VIZUALIZER_HPP
