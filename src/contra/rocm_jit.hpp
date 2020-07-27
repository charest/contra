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

namespace llvm {
class Linker;
}

namespace contra {

class ROCmJIT : public DeviceJIT {
public:
  
  ROCmJIT(utils::BuilderHelper &);

  virtual std::unique_ptr<llvm::Module> createModule(const std::string &) override;

  virtual void addModule(std::unique_ptr<llvm::Module> M) override;
  virtual void addModule(const llvm::Module * M) override;

private:
  
  llvm::BasicBlock* replacePrint2(llvm::Module &, llvm::CallInst*);

  std::unique_ptr<llvm::Module> cloneModule(const llvm::Module &);

  void runOnModule(llvm::Module &);

  std::string compile(
      llvm::Module&,
      const std::string &,
      llvm::CodeGenFileType);

  void assemble(
      llvm::Module &,
      std::string,
      bool IncludeDeviceLibs);

  std::vector<char> compileAndLink(llvm::Module &, const std::string&);
  
  std::unique_ptr<llvm::Module> splitOutReduce(llvm::Module &);

  llvm::TargetMachine * TargetMachine_ = nullptr;

  std::unique_ptr<llvm::Module> UserModule_;

  unsigned LinkFlags_ = 0;
};

} // end namespace

#endif // CONTRA_ROCM_JIT_H
