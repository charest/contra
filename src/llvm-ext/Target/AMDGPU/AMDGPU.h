//===-- AMDGPU.h - MachineFunction passes hw codegen --------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXT_TARGET_AMDGPU_AMDGPU_H
#define LLVM_EXT_TARGET_AMDGPU_AMDGPU_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;
class ModulePass;
class PassRegistry;

ModulePass *createAMDGPUPrintfRuntimeBinding();
void initializeAMDGPUPrintfRuntimeBindingPass(PassRegistry&);
extern char &AMDGPUPrintfRuntimeBindingID;

void initializeAMDGPULowerAllocaPass(PassRegistry &);
FunctionPass *createAMDGPULowerAllocaPass();

struct AMDGPUPrintfRuntimeBindingPass
    : PassInfoMixin<AMDGPUPrintfRuntimeBindingPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

}

#endif
