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
  
  std::unique_ptr<llvm::Module> cloneModule(const llvm::Module &);

  void runOnModule(llvm::Module &);

  void compile(llvm::Module&, llvm::raw_pwrite_stream&);

  void assemble(llvm::Module &, bool);

  std::vector<char> compileAndLink(llvm::Module &);

  void linkFiles(llvm::Linker &, const std::vector<std::string>&, unsigned);
  
  llvm::Instruction* replaceSync(llvm::Module &, llvm::CallInst*);
  llvm::CallInst* replacePrint(llvm::Module &, llvm::CallInst*);
  llvm::Value* replacePrint2(llvm::Module &, llvm::CallInst*);
  
  llvm::TargetMachine * TargetMachine_ = nullptr;

  std::vector<std::unique_ptr<llvm::Module>> UserModules_;

  unsigned LinkFlags_ = 0;
};

} // end namespace

#endif // CONTRA_ROCM_JIT_H
