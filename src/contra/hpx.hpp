#ifndef CONTRA_HPX_HPP
#define CONTRA_HPX_HPP

#ifdef HAVE_HPX

#include "reductions.hpp"
#include "tasking.hpp"

#include <forward_list>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Hpx Reduction
////////////////////////////////////////////////////////////////////////////////
class HpxReduceInfo : public AbstractReduceInfo {
  int Id_ = -1;

  llvm::FunctionType* ApplyT_ = nullptr;
  llvm::FunctionType* FoldT_ = nullptr;
  llvm::FunctionType* InitT_ = nullptr;
  
  std::string ApplyN_;
  std::string FoldN_;
  std::string InitN_;

  std::size_t DataSize_ = 0;

public:

  HpxReduceInfo(
      int Id,
      llvm::Function* Apply,
      llvm::Function* Fold,
      llvm::Function* Init,
      std::size_t DataSize)
    : 
      Id_(Id),
      ApplyT_(Apply->getFunctionType()),
      FoldT_(Fold->getFunctionType()),
      InitT_(Init->getFunctionType()),
      ApplyN_(Apply->getName()),
      FoldN_(Fold->getName()),
      InitN_(Init->getName()),
      DataSize_(DataSize)
  {}

  auto getId() const { return Id_; }
  
  auto getApplyType() const { return ApplyT_; }
  const auto & getApplyName() const { return ApplyN_; }

  auto getFoldType() const { return FoldT_; }
  const auto & getFoldName() const { return FoldN_; }

  auto getInitType() const { return InitT_; }
  const auto & getInitName() const { return InitN_; }

  auto getDataSize() const { return DataSize_; }

};


////////////////////////////////////////////////////////////////////////////////
// Hpx Tasker
////////////////////////////////////////////////////////////////////////////////
class HpxTasker : public AbstractTasker {
protected:
  
  const TaskInfo * TopLevelTask_ = nullptr;

public:
 
  HpxTasker(utils::BuilderHelper &);
  
  virtual ~HpxTasker();

  virtual void startRuntime(llvm::Module &) override;
  
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      llvm::Type*) override {}
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      const std::vector<llvm::Value*> & = {}) override;

  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      std::vector<llvm::Value*>,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      const AbstractReduceInfo *) override {}
  
  virtual void setTopLevelTask(
      llvm::Module &,
      const TaskInfo & TaskI) override
  { TopLevelTask_ = &TaskI; }
  
  virtual std::unique_ptr<AbstractReduceInfo> createReductionOp(
      llvm::Module &,
      const std::string &,
      const std::vector<llvm::Type*> &,
      const std::vector<ReductionType> &) override {}

  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override {}
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) override {}
  
  virtual llvm::Type* getPartitionType(llvm::Type*) const override {}

  virtual bool isPartition(llvm::Type*) const override {}
  virtual bool isPartition(llvm::Value*) const override {}
  virtual void destroyPartition(llvm::Module &, llvm::Value*) override {}
  
  virtual llvm::Type* getFieldType(llvm::Type*) const override {}

  virtual bool isField(llvm::Value*) const override {}
  virtual void createField(
      llvm::Module &,
      llvm::Value*, 
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override {}
  virtual void destroyField(llvm::Module &, llvm::Value*) override {}
  
  virtual bool isAccessor(llvm::Type*) const override {}
  virtual bool isAccessor(llvm::Value*) const override {}
  virtual llvm::Type* getAccessorType() const override {}
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value* = nullptr) const override  {}
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value* = nullptr) const override {}
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) override {}

protected:
  
};

} // namepsace

#endif // HAVE_HPX
#endif // LIBRT_HPX_HPP
