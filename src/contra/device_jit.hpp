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
  
  DeviceJIT(utils::BuilderHelper & TheHelper) :
    TheHelper_(TheHelper),
    Builder_(TheHelper.getBuilder()),
    TheContext_(TheHelper.getContext())
  {}

  virtual ~DeviceJIT() {};

  virtual std::unique_ptr<llvm::Module> createModule(
      const std::string  & = "") = 0;

  virtual void addModule(std::unique_ptr<llvm::Module> M) = 0;
  virtual void addModule(const llvm::Module * M) = 0;

protected:
  
  llvm::CallInst* replaceIntrinsic(
      llvm::Module &,
      llvm::CallInst*,
      unsigned,
      const std::vector<llvm::Type*> & = {});
  llvm::CallInst* replaceName(llvm::Module &, llvm::CallInst*, const std::string&);

  utils::BuilderHelper & TheHelper_;

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;
};

} // end namespace

#endif // CONTRA_DEVICE_JIT_H
