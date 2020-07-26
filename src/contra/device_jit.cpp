#include "device_jit.hpp"


using namespace utils;
using namespace llvm;

namespace contra {

//==============================================================================
// Helper to replace math
//==============================================================================
CallInst* DeviceJIT::replaceIntrinsic(Module &M, CallInst* CallI, unsigned Intr)
{
  std::vector<Value*> ArgVs;
  for (auto & Arg : CallI->args()) ArgVs.push_back(Arg.get());
  auto IntrinsicF = Intrinsic::getDeclaration(&M, Intr);
  auto TmpB = IRBuilder<>(TheContext_);
  return TmpB.CreateCall(IntrinsicF, ArgVs, CallI->getName());
}


} // namespace
