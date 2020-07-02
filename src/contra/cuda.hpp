#ifndef CONTRA_CUDA_HPP
#define CONTRA_CUDA_HPP

#include "config.hpp"

#include "tasking.hpp"

#include <forward_list>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Cuda Reduction
////////////////////////////////////////////////////////////////////////////////
class CudaReduceInfo : public AbstractReduceInfo {

  std::vector<llvm::Type*> VarTypes_;
  std::vector<ReductionType> ReduceTypes_;

public:

  CudaReduceInfo(
      const std::vector<llvm::Type*> & VarTypes,
      const std::vector<ReductionType> & ReduceTypes) :
    VarTypes_(VarTypes),
    ReduceTypes_(ReduceTypes)
  {}

  auto getNumReductions() const { return VarTypes_.size(); }
  
  const auto & getVarTypes() const { return VarTypes_; }
  auto getVarType(unsigned i) const { return VarTypes_[i]; }
  auto getReduceOp(unsigned i) const { return ReduceTypes_[i]; }

};


////////////////////////////////////////////////////////////////////////////////
// Cuda Tasker
////////////////////////////////////////////////////////////////////////////////

class CudaTasker : public AbstractTasker {

  llvm::StructType* FieldType_ = nullptr;
  llvm::StructType* AccessorType_ = nullptr;
  llvm::StructType* IndexSpaceType_ = nullptr;
  llvm::StructType* IndexPartitionType_ = nullptr;

  llvm::Type* PartitionInfoType_ = nullptr;

  const TaskInfo * TopLevelTask_ = nullptr;
  
  struct TaskEntry {
    llvm::AllocaInst* ResultAlloca = nullptr;
  };

  std::forward_list<TaskEntry> TaskAllocas_;

public:
 
  CudaTasker(utils::BuilderHelper & TheHelper);

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
  llvm::StructType* createFieldType();
  llvm::StructType* createAccessorType();
  llvm::StructType* createIndexPartitionType();

  llvm::AllocaInst* createPartitionInfo(llvm::Module &);
  void destroyPartitionInfo(llvm::Module &, llvm::AllocaInst*);
  
  auto & getCurrentTask() { return TaskAllocas_.front(); }
  const auto & getCurrentTask() const { return TaskAllocas_.front(); }

  auto & startTask() { 
    TaskAllocas_.push_front({});
    return getCurrentTask();
  }
  void finishTask() { TaskAllocas_.pop_front(); }
};

} // namepsace

#endif // CONTRA_CUDA_HPP
