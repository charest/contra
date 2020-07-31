#include "device_jit.hpp"


using namespace utils;
using namespace llvm;

namespace contra {

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
