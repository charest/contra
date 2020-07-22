#ifndef CONTRA_COMPILER_HPP
#define CONTRA_COMPILER_HPP

#include "config.hpp"

#include "llvm/Support/CodeGen.h"

#include <string>

namespace llvm {
class Module;
class TargetMachine;
}

namespace contra {

void compile(llvm::Module &, const std::string &);

std::string compileKernel(
    llvm::Module & TheModule,
    llvm::TargetMachine * TM,
    const std::string & Filename = "",
    llvm::CodeGenFileType FileType = llvm::CGFT_AssemblyFile);

} // namespace

#endif // CONTRA_COMPILER_HPP
