#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "llvm/IR/IRBuilder.h"

#include <iostream>
#include <string>
#include <vector>

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

public:

  TaskInfo(int Id, const std::string & Name, llvm::Function * Func)
    : Id_(Id), Name_(Name), Function_(Func), FunctionType_(Func->getFunctionType())
  {}

  auto getId() const { return Id_; }
  const auto & getName() const { return Name_; }
  auto getFunction() const { return Function_; }
  auto getFunctionType() const { return FunctionType_; }

  bool isTop() const { return IsTop_; }
  void setTop(bool IsTop = true) { IsTop_ = IsTop; }
};

//==============================================================================
// Main tasking interface
//==============================================================================
class AbstractTasker {

  unsigned IdCounter_ = 0;
  bool IsStarted_ = false;
  
  std::map<std::string, TaskInfo> TaskTable_;

protected:

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

public:
  
  AbstractTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext) :
    Builder_(TheBuilder), TheContext_(TheContext)
  {}
  
  virtual ~AbstractTasker() = default;

  struct PreambleResult {
    llvm::Function* TheFunction;
    std::vector<llvm::AllocaInst*> ArgAllocas;
  };

  // abstraact interface
  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &, llvm::Function*) = 0;
  virtual void taskPostamble(llvm::Module &, llvm::Value*) = 0;

  virtual void preregisterTask(llvm::Module &, const std::string &, const TaskInfo &) = 0;
  virtual void postregisterTask(llvm::Module &, const std::string &, const TaskInfo &) = 0;
  
  virtual void setTopLevelTask(llvm::Module &, int) = 0;
  virtual llvm::Value* startRuntime(llvm::Module &, int, char **) = 0;
  
  virtual llvm::Value* launch(llvm::Module &, const std::string &, const TaskInfo &,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &) = 0;

  virtual llvm::Value* getFuture(llvm::Module &, llvm::Value*, llvm::Type*, llvm::Value*)=0;

  // registration
  void preregisterTasks(llvm::Module &);
  void postregisterTasks(llvm::Module &);

  // startup interface
  llvm::Value* start(llvm::Module & TheModule, int Argc, char ** Argv);
  
  bool isStarted() const { return IsStarted_; }
  void setStarted() { IsStarted_ = true; }
  
  // Task table interface
  TaskInfo & insertTask(const std::string & Name, llvm::Function* F);

  bool isTask(const std::string & Name) const
  { return TaskTable_.count(Name); }

  const auto & getTask(const std::string & Name) const
  { return TaskTable_.at(Name); }

  

protected:
  
  auto getNextId() { return IdCounter_++; }

  // type helpers
  llvm::Type* reduceStruct(llvm::StructType *, const llvm::Module &) const;
  llvm::Value* sanitize(llvm::Value*, const llvm::Module &) const;
  void sanitize(std::vector<llvm::Value*> & Vs, const llvm::Module &) const;
  llvm::Value* load(llvm::Value *, const llvm::Module &, std::string) const;
  void store(llvm::Value*, llvm::AllocaInst *) const;
};

} // namespace

#endif // CONTRA_TASKING_HPP
