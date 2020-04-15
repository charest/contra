#ifndef CONTRA_LOOP_HPP
#define CONTRA_LOOP_HPP

#include "config.hpp"
#include "recursive.hpp"

#include <deque>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class LoopLifter : public RecursiveAstVisiter {

  std::deque<std::unique_ptr<FunctionAST>> FunctionQueue_;
  unsigned LoopCounter_ = 0;

  std::string CurrentName_;

  auto getNextId() { return LoopCounter_++; }
  auto makeName() { 
    std::stringstream Name;
    Name << "__" << CurrentName_ << "_loop" << getNextId() << "__";
    return Name.str();
  }

  // function queue
  void addFunctionAST( std::unique_ptr<FunctionAST> F )
  { FunctionQueue_.emplace_back( std::move(F) ); }


public:

  std::unique_ptr<FunctionAST> getNextFunctionAST()
  {
    if (FunctionQueue_.empty())
      return nullptr;
    else {
      auto F = std::move(FunctionQueue_.front());
      FunctionQueue_.pop_front();
      return F;
    }
  }

  // Codegen function
  void runVisitor(FunctionAST&e)
  {
    FunctionQueue_.clear();
    CurrentName_ = e.getName();
    e.accept(*this);
  }

  virtual void postVisit(ForeachStmtAST&e) override;

};



}

#endif // CONTRA_LOOP_HPP
