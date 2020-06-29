#ifndef CONTRA_CUDA_JIT_H
#define CONTRA_CUDA_JIT_H

#include "config.hpp"
#include "compiler.hpp"
#include "cuda_rt.hpp"
#include "errors.hpp"

#include "utils/llvm_utils.hpp"

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <set>

namespace contra {

class CudaJIT {
public:
  
  CudaJIT() 
  {
    TargetMachine_ = utils::createTargetMachine(TargetNvptxString);
    if (!TargetMachine_) 
      THROW_CONTRA_ERROR(
          "Cuda backend selected but LLVM does not support 'nvptx64'");
    contra_cuda_startup();
  }


  std::unique_ptr<llvm::Module> createModule(llvm::LLVMContext & TheContext) {
    auto NewModule = std::make_unique<llvm::Module>("devicee jit", TheContext);
    NewModule->setDataLayout(TargetMachine_->createDataLayout());
    NewModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
    return NewModule;
  }

  auto getTargetMachine() { return TargetMachine_; }
  
  void addModule(std::unique_ptr<llvm::Module> M) {
    auto KernelStr = compileKernel(
        *M,
        TargetMachine_);
    contra_cuda_register_kernel(KernelStr.c_str());
  }
  
  void addModule(const llvm::Module * M) {
      std::set<llvm::GlobalValue *> ClonedDefsInSrc;
      llvm::ValueToValueMapTy VMap;
      auto ClonedModule = CloneModule(
          *M,
          VMap,
          [](const llvm::GlobalValue *GV) { return true; });

      ClonedModule->setSourceFileName("device jit");
      ClonedModule->setDataLayout(TargetMachine_->createDataLayout());
      ClonedModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
      addModule(std::move(ClonedModule));
  }


private:

  llvm::TargetMachine * TargetMachine_ = nullptr;
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
