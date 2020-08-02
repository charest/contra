#ifndef CONTRA_ROCM_HPP
#define CONTRA_ROCM_HPP

#include "config.hpp"

#include "tasking.hpp"

#include <forward_list>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// ROCm Reduction
////////////////////////////////////////////////////////////////////////////////
class ROCmReduceInfo : public AbstractReduceInfo {

  std::vector<llvm::Type*> VarTypes_;
  std::vector<ReductionType> ReduceTypes_;
  
  std::string InitN_,  InitPtrN_;
  llvm::FunctionType * InitT_;

  std::string ApplyN_, ApplyPtrN_;
  llvm::FunctionType * ApplyT_;
  
  std::string FoldN_,  FoldPtrN_;
  llvm::FunctionType * FoldT_;

public:

  ROCmReduceInfo(
      const std::vector<llvm::Type*> & VarTypes,
      const std::vector<ReductionType> & ReduceTypes,
      llvm::Function * InitF,
      const std::string & InitPtrN,
      llvm::Function * ApplyF,
      const std::string & ApplyPtrN,
      llvm::Function * FoldF,
      const std::string & FoldPtrN) :
    VarTypes_(VarTypes),
    ReduceTypes_(ReduceTypes),
    InitN_(InitF->getName()),
    InitPtrN_(InitPtrN),
    InitT_(InitF->getFunctionType()),
    ApplyN_(ApplyF->getName()),
    ApplyPtrN_(ApplyPtrN),
    ApplyT_(ApplyF->getFunctionType()),
    FoldN_(FoldF->getName()),
    FoldPtrN_(FoldPtrN),
    FoldT_(FoldF->getFunctionType())
  {}

  auto getNumReductions() const { return VarTypes_.size(); }
  
  const auto & getVarTypes() const { return VarTypes_; }
  auto getVarType(unsigned i) const { return VarTypes_[i]; }
  auto getReduceOp(unsigned i) const { return ReduceTypes_[i]; }

  const auto & getInitName() const { return InitN_; }
  const auto & getInitPtrName() const { return InitPtrN_; }
  auto getInitType() const { return InitT_; }

  const auto & getApplyName() const { return ApplyN_; }
  const auto & getApplyPtrName() const { return ApplyPtrN_; }
  auto getApplyType() const { return ApplyT_; }

  const auto & getFoldName() const { return FoldN_; }
  const auto & getFoldPtrName() const { return FoldPtrN_; }
  auto getFoldType() const { return FoldT_; }
};


////////////////////////////////////////////////////////////////////////////////
// ROCm Tasker
////////////////////////////////////////////////////////////////////////////////

class ROCmTasker : public AbstractTasker {

  llvm::Type* Int64Type_ = nullptr;
  llvm::Type* UInt32Type_ = nullptr;
  llvm::Type* UInt64Type_ = nullptr;

  llvm::StructType* Dim3Type_ = nullptr;
  llvm::StructType* ReducedDim3Type_ = nullptr;
  llvm::StructType* StreamType_ = nullptr;

  llvm::StructType* FieldType_ = nullptr;
  llvm::StructType* AccessorType_ = nullptr;
  llvm::StructType* IndexSpaceType_ = nullptr;
  llvm::StructType* IndexPartitionType_ = nullptr;

  llvm::Type* PartitionInfoType_ = nullptr;
  llvm::Type* TaskInfoType_ = nullptr;

  const TaskInfo * TopLevelTask_ = nullptr;
  
  struct TaskEntry {
    llvm::BasicBlock* MergeBlock = nullptr;
    llvm::AllocaInst* TaskInfoAlloca = nullptr;
    bool HasPrintf = false;
    
    llvm::AllocaInst* ResultThreadAlloca = nullptr;
    llvm::AllocaInst* ResultBlockAlloca = nullptr;
    llvm::AllocaInst* IndexSizeAlloca = nullptr;
  };

  std::forward_list<TaskEntry> TaskAllocas_;

public:
 
  ROCmTasker(utils::BuilderHelper & TheHelper);

  virtual void startRuntime(
      llvm::Module &,
      int,
      char **) override;
  
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      llvm::Function*) override;
  
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      llvm::Type*) override;
  
  virtual void taskPostamble(
      llvm::Module &,
      llvm::Value*,
      bool) override;
  
  using AbstractTasker::launch;
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      std::vector<llvm::Value*>,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      const AbstractReduceInfo*) override;
  
  virtual void setTopLevelTask(
      llvm::Module &,
      const TaskInfo & TaskI) override
  { TopLevelTask_ = &TaskI; }
  
  virtual std::unique_ptr<AbstractReduceInfo> createReductionOp(
      llvm::Module &,
      const std::string &,
      const std::vector<llvm::Type*> &,
      const std::vector<ReductionType> &) override;

  
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) override;
  
  virtual llvm::Type* getPartitionType(llvm::Type*) const override
  { return IndexPartitionType_; }

  virtual bool isPartition(llvm::Type*) const override;
  virtual bool isPartition(llvm::Value*) const override;
  virtual void destroyPartition(llvm::Module &, llvm::Value*) override;
  
  virtual llvm::Type* getFieldType(llvm::Type*) const override
  { return FieldType_; }

  virtual bool isField(llvm::Value*) const override;
  virtual void createField(
      llvm::Module &,
      llvm::Value*, 
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override;
  virtual void destroyField(llvm::Module &, llvm::Value*) override;
  
  virtual bool isAccessor(llvm::Type*) const override;
  virtual bool isAccessor(llvm::Value*) const override;
  virtual llvm::Type* getAccessorType() const override
  { return AccessorType_; }
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value* = nullptr) const override;
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value* = nullptr) const override;
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) override;

protected:

  llvm::StructType* createDim3Type();
  llvm::StructType* createReducedDim3Type();
  llvm::StructType* createStreamType();

  llvm::StructType* createFieldType();
  llvm::StructType* createAccessorType();
  llvm::StructType* createIndexPartitionType();

  llvm::AllocaInst* createPartitionInfo(llvm::Module &);
  void destroyPartitionInfo(llvm::Module &, llvm::AllocaInst*);
  
  llvm::AllocaInst* createTaskInfo(llvm::Module &);
  void destroyTaskInfo(llvm::Module &, llvm::AllocaInst*);

  void createIndexSpaceFromPartition(
      llvm::Value*,
      llvm::AllocaInst*,
      llvm::AllocaInst*);
  
  auto & getCurrentTask() { return TaskAllocas_.front(); }
  const auto & getCurrentTask() const { return TaskAllocas_.front(); }

  auto & startTask() { 
    TaskAllocas_.push_front({});
    return getCurrentTask();
  }
  void finishTask() { TaskAllocas_.pop_front(); }

  llvm::Value* getThreadID(llvm::Module &) const;

  std::pair<llvm::Value*, llvm::Value*> offsetAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) const;
};

} // namepsace

#endif // CONTRA_ROCM_HPP
