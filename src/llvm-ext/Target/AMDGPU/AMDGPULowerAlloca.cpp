//===----------------------------------------------------------------------===//
//
// For all alloca instructions, and add a pair of cast to local address for
// each of them. For example,
//
//   %A = alloca i32
//   store i32 0, i32* %A ; emits st.u32
//
// will be transformed to
//
//   %A = alloca i32
//   %Local = addrspacecast i32* %A to i32 addrspace(5)*
//   %Generic = addrspacecast i32 addrspace(5)* %A to i32*
//   store i32 0, i32 addrspace(5)* %Generic ; emits st.local.u32
//
// And we will rely on AMDGPUInferAddressSpaces to combine the last two
// instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support//raw_ostream.h"

using namespace llvm;

namespace llvm {
void initializeAMDGPULowerAllocaPass(PassRegistry &);
}

namespace {
class AMDGPULowerAlloca : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID; // Pass identification, replacement for typeid
  enum { 
    ADDRESS_SPACE_GENERIC = 0,
    ADDRESS_SPACE_LOCAL = 5
  };
  AMDGPULowerAlloca() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "convert address space of alloca'ed memory to local";
  }
};
} // namespace

char AMDGPULowerAlloca::ID = 1;

INITIALIZE_PASS(AMDGPULowerAlloca, "amdgpu-lower-alloca",
                "Lower Alloca", false, false)

// =============================================================================
// Main function for this pass.
// =================gg============================================================
bool AMDGPULowerAlloca::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  bool Changed = false;
  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto allocaInst = dyn_cast<AllocaInst>(&I)) {
        auto PTy = cast<PointerType>(allocaInst->getType());
        auto ETy = allocaInst->getAllocatedType();
        if (PTy->getAddressSpace() == ADDRESS_SPACE_LOCAL) continue;
        Changed = true;
        auto LocalAddrTy = PointerType::get(ETy, ADDRESS_SPACE_LOCAL);
        auto NewASCToLocal = new AddrSpaceCastInst(allocaInst, LocalAddrTy, "");
        auto GenericAddrTy = PointerType::get(ETy, ADDRESS_SPACE_GENERIC);
        auto NewASCToGeneric =
            new AddrSpaceCastInst(NewASCToLocal, GenericAddrTy, "");
        NewASCToLocal->insertAfter(allocaInst);
        NewASCToGeneric->insertAfter(NewASCToLocal);
        for (Value::use_iterator UI = allocaInst->use_begin(),
                                 UE = allocaInst->use_end();
             UI != UE;) {
          // Check Load, Store, GEP, and BitCast Uses on alloca and make them
          // use the converted generic address, in order to expose non-generic
          // addrspacecast to AMDGPUInferAddressSpaces. For other types
          // of instructions this is unnecessary and may introduce redundant
          // address cast.
          const auto &AllocaUse = *UI++;
          auto LI = dyn_cast<LoadInst>(AllocaUse.getUser());
          if (LI && LI->getPointerOperand() == allocaInst &&
              !LI->isVolatile()) {
            LI->setOperand(LI->getPointerOperandIndex(), allocaInst);
            continue;
          }
          auto SI = dyn_cast<StoreInst>(AllocaUse.getUser());
          if (SI && SI->getPointerOperand() == allocaInst &&
              !SI->isVolatile()) {
            SI->setOperand(SI->getPointerOperandIndex(), allocaInst);
            continue;
          }
          auto GI = dyn_cast<GetElementPtrInst>(AllocaUse.getUser());
          if (GI && GI->getPointerOperand() == allocaInst) {
            GI->setOperand(GI->getPointerOperandIndex(), allocaInst);
            continue;
          }
          auto BI = dyn_cast<BitCastInst>(AllocaUse.getUser());
          if (BI && BI->getOperand(0) == allocaInst) {
            BI->setOperand(0, allocaInst);
            continue;
          }
        }
      }
    }
  return Changed;
}

FunctionPass *llvm::createAMDGPULowerAllocaPass() {
  return new AMDGPULowerAlloca();
}
