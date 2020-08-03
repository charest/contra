#include "device_jit.hpp"

#include "args.hpp"

using namespace utils;
using namespace llvm;

namespace contra {

//==============================================================================
// Options
//==============================================================================
llvm::cl::opt<std::string> OptionTargetCPU(
    "mcpu",
    llvm::cl::desc("Target CPU"),
    llvm::cl::cat(OptionCategory));

llvm::cl::opt<int> OptionMaxBlockSize(
    "max-block-size",
    llvm::cl::desc(
      "The maximum block size to use with GPU-based backends. "
      "If unset, this information is extracted from the particular device."
    ),
    llvm::cl::cat(OptionCategory));

  
//==============================================================================
// Constructor
//==============================================================================
DeviceJIT::DeviceJIT(utils::BuilderHelper & TheHelper) :
  TheHelper_(TheHelper),
  Builder_(TheHelper.getBuilder()),
  TheContext_(TheHelper.getContext())
{
  if (!OptionTargetCPU.empty()) TargetCPU_ = OptionTargetCPU;
  if (OptionMaxBlockSize) MaxBlockSize_ = OptionMaxBlockSize;
}

//==============================================================================
// Helper to replace math
//==============================================================================
CallInst* DeviceJIT::replaceIntrinsic(
    Module &M,
    CallInst* CallI,
    unsigned Intr,
    const std::vector<Type*> & Tys)
{
  std::vector<Value*> ArgVs;
  for (auto & Arg : CallI->args()) ArgVs.push_back(Arg.get());
  auto IntrinsicF = Intrinsic::getDeclaration(&M, Intr, Tys);
  IRBuilder<> TmpB(TheContext_);
  return TmpB.CreateCall(IntrinsicF, ArgVs, CallI->getName());
}

//==============================================================================
// Helper to replace print function
//==============================================================================
CallInst* DeviceJIT::replaceName(
    Module &M,
    CallInst* CallI,
    const std::string & Name)
{
  std::vector<Value*> ArgVs;
  for (auto & Arg : CallI->args()) ArgVs.push_back(Arg.get());

  auto FTy = CallI->getFunctionType();
  
  Function * NewF = M.getFunction(Name);

  if (!NewF) {  
    auto F = CallI->getFunction();
    NewF = Function::Create(
        FTy,
        F->getLinkage(),
        F->getAddressSpace(),
        Name,
        &M);
  }
  
  // create new instruction            
  IRBuilder<> TmpB(TheContext_);
  
  auto RetT = NewF->getReturnType();
  if (!RetT || RetT->isVoidTy())
    return TmpB.CreateCall(NewF, ArgVs);
  else
    return TmpB.CreateCall(NewF, ArgVs, CallI->getName());
}

//==============================================================================
// Calls function
//==============================================================================
bool DeviceJIT::callsFunction(Module & M, const std::string & Name)
{
  for (auto & F : M)
    for (auto & BB : F)
      for (auto & I : BB)
        if (auto CallI = dyn_cast<CallInst>(&I)) {
          auto CallF = CallI->getCalledFunction();
          if (CallF->getName() == Name) return
            true;
        }
  return false;
}

} // namespace
