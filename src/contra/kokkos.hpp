#ifndef CONTRA_KOKKOS_HPP
#define CONTRA_KOKKOS_HPP

#include "config.hpp"

#ifdef HAVE_KOKKOS

#include "tasking.hpp"

namespace contra {

class KokkosTasker : public AbstractTasker {

  llvm::StructType* FieldDataType_ = nullptr;

  const TaskInfo * TopLevelTask_ = nullptr;

public:
 
  KokkosTasker(utils::BuilderHelper & TheHelper);

  virtual void startRuntime(
      llvm::Module &,
      int,
      char **) override;
  virtual void stopRuntime(llvm::Module &) override;
  
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &,
      llvm::Type*) override;

  using AbstractTasker::launch;
  
  virtual llvm::Value* launch(
      llvm::Module &,
      const TaskInfo &,
      std::vector<llvm::Value*>,
      const std::vector<llvm::Value*> &,
      llvm::Value*,
      const AbstractReduceInfo *) override
  { return nullptr; }
  
  virtual void setTopLevelTask(
      llvm::Module &,
      const TaskInfo & TaskI) override
  { TopLevelTask_ = &TaskI; }
  
  virtual std::unique_ptr<AbstractReduceInfo> createReductionOp(
      llvm::Module &,
      const std::string &,
      const std::vector<llvm::Type*> &,
      const std::vector<ReductionType> &) override
  { return nullptr; }

  
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override { return nullptr; }
  virtual llvm::AllocaInst* createPartition(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*) override { return nullptr; }
  
  virtual llvm::Type* getPartitionType(llvm::Type*) const override
  {return nullptr;}
  virtual bool isPartition(llvm::Type*) const override { return false; }
  virtual bool isPartition(llvm::Value*) const override { return false; }
  virtual void destroyPartition(llvm::Module &, llvm::Value*) override {}
  
  
  virtual llvm::Type* getFieldType(llvm::Type*) const override
  { return FieldDataType_; }

  virtual bool isField(llvm::Value*) const override;
  virtual void createField(
      llvm::Module &,
      llvm::Value*, 
      const std::string &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value*) override;
  virtual void destroyField(llvm::Module &, llvm::Value*) override;
  
  virtual bool isAccessor(llvm::Type*) const override {return false;}
  virtual bool isAccessor(llvm::Value*) const override {return false;}
  virtual llvm::Type* getAccessorType() const override {return nullptr;}
  virtual void storeAccessor(
      llvm::Module &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value* = nullptr) const override {};
  virtual llvm::Value* loadAccessor(
      llvm::Module &,
      llvm::Type*,
      llvm::Value*,
      llvm::Value* = nullptr) const override {return nullptr;};
  virtual void destroyAccessor(llvm::Module &, llvm::Value*) override {};
  
protected:
  llvm::StructType* createFieldDataType();
};

} // namepsace

#endif // HAVE_KOKKOS
#endif // LIBRT_LEGION_HPP
