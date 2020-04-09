#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "taskinfo.hpp"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace contra {

//==============================================================================
// Main tasking interface
//==============================================================================
class AbstractTasker {

  unsigned IdCounter_ = 0;
  bool IsStarted_ = false;
  
  std::map<std::string, TaskInfo> TaskTable_;
  std::map<std::string, llvm::Value*> FutureTable_;

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
    llvm::AllocaInst* Index;
  };

  // abstraact interface
  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &, llvm::Function*) = 0;
  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &,
      const std::vector<std::string> &, const std::vector<llvm::Type*> &, bool = false) = 0;
  virtual void taskPostamble(llvm::Module &, llvm::Value* = nullptr) = 0;

  virtual void preregisterTask(llvm::Module &, const std::string &, const TaskInfo &) = 0;
  virtual void postregisterTask(llvm::Module &, const std::string &, const TaskInfo &) = 0;
  
  virtual void setTopLevelTask(llvm::Module &, int) = 0;
  virtual llvm::Value* startRuntime(llvm::Module &, int, char **) = 0;
  
  virtual llvm::Value* launch(llvm::Module &, const std::string &, int,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &) = 0;
  virtual llvm::Value* launch(llvm::Module &, const std::string &, int,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &,
      llvm::Value*, llvm::Value*) = 0;

  virtual bool isFuture(llvm::Value*) const = 0;
  virtual llvm::Value* createFuture(llvm::Module &,llvm::Function*, const std::string &) = 0;
  virtual llvm::Value* loadFuture(llvm::Module &, llvm::Value*, llvm::Type*, llvm::Value*)=0;
  virtual void destroyFuture(llvm::Module &, llvm::Value*) = 0;

  // registration
  void preregisterTasks(llvm::Module &);
  void postregisterTasks(llvm::Module &);

  // startup interface
  llvm::Value* start(llvm::Module & TheModule, int Argc, char ** Argv);
  
  bool isStarted() const { return IsStarted_; }
  void setStarted() { IsStarted_ = true; }
  
  // Task table interface
  TaskInfo & insertTask(const std::string & Name, llvm::Function* F);
  TaskInfo & insertTask(const std::string & Name);

  bool isTask(const std::string & Name) const
  { return TaskTable_.count(Name); }

  TaskInfo popTask(const std::string & Name);

  const auto & getTask(const std::string & Name) const
  { return TaskTable_.at(Name); }

  // future table interface
  void createFuture(const std::string & Name, llvm::Value* Alloca)
  { FutureTable_[Name] = Alloca; }

  llvm::Value* getFuture(const std::string & Name)
  { return FutureTable_.at(Name); }

  llvm::Value* popFuture(const std::string & Name);

  bool isFuture(const std::string & Name)
  { return FutureTable_.count(Name); }

  void destroyFutures(llvm::Module &, const std::set<std::string> &);
  

protected:
  
  auto getNextId() { return IdCounter_++; }

  // helpers
  llvm::Type* reduceStruct(llvm::StructType *, const llvm::Module &) const;
  llvm::Type* reduceArray(llvm::ArrayType *, const llvm::Module &) const;
  llvm::Value* sanitize(llvm::Value*, const llvm::Module &) const;
  void sanitize(std::vector<llvm::Value*> & Vs, const llvm::Module &) const;
  llvm::Value* load(llvm::Value *, const llvm::Module &, std::string) const;
  void store(llvm::Value*, llvm::AllocaInst *) const;

  llvm::Value* offsetPointer(llvm::AllocaInst* PointerA, llvm::AllocaInst* OffsetA,
      const std::string & Name = "");
    
  void increment(llvm::Value* OffsetA, llvm::Value* IncrV,
      const std::string & Name = "");
   
  void memCopy(llvm::Value* SrcGEP, llvm::AllocaInst* TgtA, llvm::Value* SizeV, 
      const std::string & Name = "");

  llvm::Value* accessStructMember(llvm::AllocaInst* StructA, int i,
      const std::string & Name = "");
  
  llvm::Value* loadStructMember(llvm::AllocaInst* StructA, int i,
      const std::string & Name = "");
  
  void storeStructMember(llvm::Value* ValueV, llvm::AllocaInst* StructA, int i,
      const std::string & Name = "");
};

} // namespace

#endif // CONTRA_TASKING_HPP
