#ifndef CONTRA_ROCM_JIT_H
#define CONTRA_ROCM_JIT_H

#include "config.hpp"
#include "device_jit.hpp"
#include "jit.hpp"

#include "utils/builder.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>

namespace contra {

class ROCmJIT : public DeviceJIT {
public:
  
  ROCmJIT(utils::BuilderHelper &);

  virtual std::unique_ptr<llvm::Module> createModule() override;

  virtual void addModule(std::unique_ptr<llvm::Module> M) override;
  virtual void addModule(const llvm::Module * M) override;

private:

  llvm::CallInst* replacePrint(llvm::Module &, llvm::CallInst*);

  void runOnModule(llvm::Module &);

  std::string compile(
      llvm::Module&,
      const std::string &,
      llvm::CodeGenFileType);

  std::unique_ptr<llvm::Module> insertBitcode(
      std::unique_ptr<llvm::Module>,
      std::string);

  std::vector<char> compileAndLink(llvm::Module &, const std::string&);

  llvm::TargetMachine * TargetMachine_ = nullptr;
};

} // end namespace

#endif // CONTRA_ROCM_JIT_H
