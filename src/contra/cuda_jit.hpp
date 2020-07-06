#ifndef CONTRA_CUDA_JIT_H
#define CONTRA_CUDA_JIT_H

#include "config.hpp"
#include "device_jit.hpp"

#include "utils/builder.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>

namespace contra {

class CudaJIT : public DeviceJIT {
public:
  
  CudaJIT(utils::BuilderHelper &);
  ~CudaJIT();

  virtual std::unique_ptr<llvm::Module> createModule() override;

  virtual void addModule(std::unique_ptr<llvm::Module> M) override;
  virtual void addModule(const llvm::Module * M) override;


private:

  llvm::CallInst* replacePrint(llvm::Module &, llvm::CallInst*);
  llvm::CallInst* replaceIntrinsic(llvm::Module &, llvm::CallInst*, unsigned);

  llvm::TargetMachine * TargetMachine_ = nullptr;
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
