#ifndef CONTRA_KOKKOS_HPP
#define CONTRA_KOKKOS_HPP

#include "config.hpp"

#ifdef HAVE_KOKKOS

#include "tasking.hpp"

namespace contra {

class KokkosTasker : public AbstractTasker {

  llvm::StructType* IndexSpaceDataType_ = nullptr;
  llvm::StructType* FieldDataType_ = nullptr;

  const TaskInfo * TopLevelTask_ = nullptr;

public:
 
  KokkosTasker(utils::BuilderHelper & TheHelper);

  virtual llvm::Value* startRuntime(
      llvm::Module &,
      int,
      char **) override;
  virtual void stopRuntime(llvm::Module &) override;
  
  virtual PreambleResult taskPreamble(
      llvm::Module &,
      const std::string &,
      const std::vector<std::string> &,
      const std::vector<llvm::Type*> &) override;
  
  virtual void setTopLevelTask(
      llvm::Module &,
      const TaskInfo & TaskI) override
  { TopLevelTask_ = &TaskI; }
  
  virtual bool isRange(llvm::Type*) const override;
  virtual bool isRange(llvm::Value*) const override;
  virtual llvm::AllocaInst* createRange(
      llvm::Module &,
      const std::string &,
      llvm::Value*,
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::Value* getRangeSize(llvm::Value*) override;
  virtual llvm::Value* getRangeStart(llvm::Value*) override;
  virtual llvm::Value* getRangeEnd(llvm::Value*) override;
  virtual llvm::Value* getRangeEndPlusOne(llvm::Value*) override;
  virtual llvm::Value* getRangeStep(llvm::Value*) override;
  virtual llvm::Value* loadRangeValue(
      llvm::Value*,
      llvm::Value*) override;
  virtual llvm::Type* getRangeType(llvm::Type*) const override
  { return IndexSpaceDataType_; }
  
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
  
protected:
  llvm::StructType* createIndexSpaceDataType();
  llvm::StructType* createFieldDataType();
};

} // namepsace

#endif // HAVE_KOKKOS
#endif // LIBRT_LEGION_HPP
