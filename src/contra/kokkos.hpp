#ifndef CONTRA_KOKKOS_HPP
#define CONTRA_KOKKOS_HPP

#include "config.hpp"

#ifdef HAVE_KOKKOS

#include "tasking.hpp"

namespace contra {

class KokkosTasker : public AbstractTasker {

  llvm::StructType* IndexSpaceDataType_ = nullptr;

public:
 
  KokkosTasker(utils::BuilderHelper & TheHelper);

  virtual llvm::Value* startRuntime(
      llvm::Module &,
      int,
      char **) override;
  virtual void stopRuntime(llvm::Module &) override;
  
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
  virtual llvm::Type* getRangeType() const override
  { return IndexSpaceDataType_; }

protected:
  llvm::StructType* createIndexSpaceDataType();
};

} // namepsace

#endif // HAVE_KOKKOS
#endif // LIBRT_LEGION_HPP
