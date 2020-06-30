#ifndef CONTRA_CUDA_JIT_H
#define CONTRA_CUDA_JIT_H

#include "config.hpp"

#include "utils/builder.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>

namespace contra {

class CudaJIT {
public:
  
  CudaJIT(utils::BuilderHelper &);
  ~CudaJIT();

  std::unique_ptr<llvm::Module> createModule();

  auto getTargetMachine() { return TargetMachine_; }
  
  void addModule(std::unique_ptr<llvm::Module> M);
  void addModule(const llvm::Module * M);


private:

  llvm::CallInst* replacePrint(llvm::Module &, llvm::CallInst*);

  utils::BuilderHelper & TheHelper_;

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;


  llvm::TargetMachine * TargetMachine_ = nullptr;
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
