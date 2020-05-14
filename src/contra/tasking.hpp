#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "serializer.hpp"
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

protected:

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

  Serializer DefaultSerializer_;
  std::map<llvm::Type*, std::unique_ptr<Serializer>> Serializer_;

public:
  
  AbstractTasker(
      llvm::IRBuilder<> & TheBuilder,
      llvm::LLVMContext & TheContext) :
    Builder_(TheBuilder),
    TheContext_(TheContext),
    DefaultSerializer_(TheBuilder, TheContext)
  {}
  
  virtual ~AbstractTasker() = default;

  struct PreambleResult {
    llvm::Function* TheFunction;
    std::vector<llvm::AllocaInst*> ArgAllocas;
    llvm::AllocaInst* Index;
  };

  // abstraact interface
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      llvm::Function*) = 0;

  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      bool,
      const std::vector<bool> & = {}) = 0;

  virtual void taskPostamble(
      llvm::Module &,
      llvm::Value* = nullptr) = 0;

  virtual void preregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) = 0;

  virtual void postregisterTask(
      llvm::Module &,
      const std::string &,
      const TaskInfo &) = 0;
  
  virtual void setTopLevelTask(llvm::Module &, int) = 0;
  virtual llvm::Value* startRuntime(llvm::Module &, int, char **) = 0;
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const std::string &,
      int,
      const std::vector<llvm::Value*> &) = 0;

  virtual llvm::Value* launch(
      llvm::Module &,
      const std::string &,
      int,
      const std::vector<llvm::Value*> &,
      llvm::Value*) = 0;

  virtual bool isFuture(llvm::Value*) const = 0;

  virtual llvm::AllocaInst* createFuture(
      llvm::Module &,
      llvm::Function*,
      const std::string &) = 0;

  virtual llvm::Value* loadFuture(
      llvm::Module &,
      llvm::Value*,
      llvm::Type*)=0;
  virtual void destroyFuture(llvm::Module &, llvm::Value*) = 0;
  virtual void toFuture(llvm::Module &, llvm::Value*, llvm::Value*) = 0;
  virtual void copyFuture(llvm::Module &, llvm::Value*, llvm::Value*) = 0;

  virtual bool isField(llvm::Value*) const = 0;
  virtual llvm::AllocaInst* createField(
      llvm::Module &,
      llvm::Function*,
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) = 0;
  virtual void destroyField(llvm::Module &, llvm::Value*) = 0;
  
  virtual bool isRange(llvm::Type*) const = 0;
  virtual bool isRange(llvm::Value*) const = 0;
  virtual llvm::Type* getRangeType() const = 0;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      llvm::Function*,
      const std::string &,
      llvm::Value*,
      llvm::Value*) = 0;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      llvm::Function*,
      llvm::Value*,
      const std::string & = "") = 0;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      llvm::Function*,
      llvm::Type*,
      llvm::Value*,
      const std::string & = "") = 0;
  virtual void destroyRange(llvm::Module &, llvm::Value*) = 0;
  virtual llvm::Value* getRangeSize(
      llvm::Module &,
      llvm::Value*) = 0;
  virtual llvm::Value* getRangeStart(
      llvm::Module &,
      llvm::Value*) = 0;
  virtual llvm::Value* getRangeEnd(
      llvm::Module &,
      llvm::Value*) = 0;
  virtual llvm::Value* loadRangeValue(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) = 0;

  virtual bool isAccessor(llvm::Type*) const = 0;
  virtual bool isAccessor(llvm::Value*) const = 0;
  virtual llvm::Type* getAccessorType() const = 0;
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value* = nullptr) const = 0;
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value* = nullptr) const = 0;
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) = 0;
  
  virtual llvm::AllocaInst* partition(
      llvm::Module &,
      llvm::Function*,
      llvm::Value*,
      llvm::Value*) = 0;
  virtual llvm::AllocaInst* partition(
      llvm::Module &,
      llvm::Function*,
      llvm::Value*,
      llvm::Type*,
      llvm::Value*,
      bool) = 0;

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

  // future interface
  void destroyFutures(llvm::Module &, const std::vector<llvm::Value*> &);

  // field interface
  void destroyFields(llvm::Module &, const std::vector<llvm::Value*> &);

  // range interface
  void destroyRanges(llvm::Module &, const std::vector<llvm::Value*> &);
  
  // accessor interface
  void destroyAccessors(llvm::Module &, const std::vector<llvm::Value*> &);


protected:
  
  auto getNextId() { return IdCounter_++; }

  // helpers
  llvm::Type* reduceStruct(llvm::StructType *, const llvm::Module &) const;
  llvm::Type* reduceArray(llvm::ArrayType *, const llvm::Module &) const;
  llvm::Value* sanitize(llvm::Value*, const llvm::Module &) const;
  void sanitize(std::vector<llvm::Value*> & Vs, const llvm::Module &) const;
  llvm::Value* load(llvm::Value *, const llvm::Module &, std::string) const;
  void store(llvm::Value*, llvm::Value *) const;

  llvm::Value* offsetPointer(
      llvm::AllocaInst* PointerA,
      llvm::AllocaInst* OffsetA,
      const std::string & Name = "");
    
  void increment(
      llvm::Value* OffsetA,
      llvm::Value* IncrV,
      const std::string & Name = "");
   
  void memCopy(
      llvm::Value* SrcGEP,
      llvm::AllocaInst* TgtA,
      llvm::Value* SizeV, 
      const std::string & Name = "");

  llvm::Value* accessStructMember(
      llvm::AllocaInst* StructA,
      int i,
      const std::string & Name = "");
  
  llvm::Value* loadStructMember(
      llvm::AllocaInst* StructA,
      int i,
      const std::string & Name = "");
  
  void storeStructMember(
      llvm::Value* ValueV,
      llvm::AllocaInst* StructA,
      int i,
      const std::string & Name = "");

  // Serializer
  llvm::Value* getSize(llvm::Value*, llvm::Type*);
  llvm::Value* serialize(llvm::Value*, llvm::Value*, llvm::Value* = nullptr);
  llvm::Value* deserialize(llvm::AllocaInst*, llvm::Value*, llvm::Value* = nullptr);

};

} // namespace

#endif // CONTRA_TASKING_HPP
