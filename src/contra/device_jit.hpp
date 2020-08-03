#ifndef CONTRA_DEVICE_JIT_H
#define CONTRA_DEVICE_JIT_H

#include "config.hpp"

#include "utils/builder.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>

namespace contra {

class DeviceJIT {
public:
  
  DeviceJIT(utils::BuilderHelper & TheHelper);

  virtual ~DeviceJIT() {};

  virtual std::unique_ptr<llvm::Module> createModule(
      const std::string  & = "") = 0;

  virtual void addModule(std::unique_ptr<llvm::Module> M) = 0;
  virtual void addModule(const llvm::Module * M) = 0;

  llvm::CallInst* replaceIntrinsic(
      llvm::Module &,
      llvm::CallInst*,
      unsigned,
      const std::vector<llvm::Type*> & = {});
  llvm::CallInst* replaceName(llvm::Module &, llvm::CallInst*, const std::string&);

  bool callsFunction(llvm::Module &, const std::string &);

  const auto & getTargetCPU() const { return TargetCPU_; }
  auto hasTargetCPU() const { return !TargetCPU_.empty(); }

  auto getMaxBlockSize() const { return MaxBlockSize_; }
  auto hasMaxBlockSize() const { return MaxBlockSize_ != 0; }

protected:
  
  utils::BuilderHelper & TheHelper_;

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

private:

  std::string TargetCPU_;
  unsigned MaxBlockSize_ = 0;
};

} // end namespace

#endif // CONTRA_DEVICE_JIT_H
