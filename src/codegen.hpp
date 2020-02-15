#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Value.h"

#include <map>
#include <memory>
#include <string>

namespace contra {

using Function = llvm::Function;
using Value = llvm::Value;

class PrototypeAST;

class CodeGen {

public:

  llvm::LLVMContext TheContext;
  llvm::IRBuilder<> Builder;
  std::unique_ptr<llvm::Module> TheModule;
  std::map<std::string, Value *> NamedValues;
  std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM;
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

  CodeGen () : Builder(TheContext) {}

  Function *getFunction(std::string Name); 
  
};

} // namespace

#endif // CONTRA_CODEGEN_HPP
