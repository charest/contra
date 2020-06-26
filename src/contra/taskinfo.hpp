#ifndef CONTRA_TASKINFO_HPP
#define CONTRA_TASKINFO_HPP

#include "reduceinfo.hpp"

#include "llvm/IR/IRBuilder.h"

#include <string>

namespace contra {

//==============================================================================
// Task info
//==============================================================================
class TaskInfo {
  int Id_ = -1;
  std::string Name_;
  llvm::Function * Function_ = nullptr;
  llvm::FunctionType * FunctionType_ = nullptr;
  bool IsTop_ = false;
  bool IsLeaf_ = false;

  std::unique_ptr<AbstractReduceInfo> Redop_;

public:

  TaskInfo(
      int Id,
      const std::string & Name,
      llvm::Function * Func) :
    Id_(Id), 
    Name_(Name),
    Function_(Func),
    FunctionType_(Func->getFunctionType())
  {}

  TaskInfo(int Id) : Id_(Id)
  {}

  auto getId() const { return Id_; }
  const auto & getName() const { return Name_; }
  
  auto getFunction() const { return Function_; }
  auto getFunctionType() const { return FunctionType_; }
  auto getReturnType() const { return FunctionType_->getReturnType(); }

  bool isTop() const { return IsTop_; }
  void setTop(bool IsTop = true) { IsTop_ = IsTop; }

  bool isLeaf() const { return IsLeaf_; }
  void setLeaf(bool IsLeaf = true) { IsLeaf_ = IsLeaf; }

  bool hasReduction() const { return static_cast<bool>(Redop_); }
  auto getReduction() const { return Redop_.get(); }
  void setReduction(std::unique_ptr<AbstractReduceInfo> Redop)
  { Redop_ = std::move(Redop); }

};

} // namespace

#endif // CONTRA_TASKINFO_HPP
