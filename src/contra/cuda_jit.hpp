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

  virtual std::unique_ptr<llvm::Module> createModule(
    const std::string & ) override;

  virtual void addModule(std::unique_ptr<llvm::Module> M) override;
  virtual void addModule(const llvm::Module * M) override;


private:

  std::string compile(
      llvm::Module & TheModule,
      const std::string & Filename = "",
      llvm::CodeGenFileType FileType = llvm::CGFT_AssemblyFile);


  llvm::CallInst* replacePrint(llvm::Module &, llvm::CallInst*);

  llvm::TargetMachine * TargetMachine_ = nullptr;
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
