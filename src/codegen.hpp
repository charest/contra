#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "debug.hpp"
#include "jit.hpp"
#include "vartype.hpp"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"

#include <map>
#include <memory>
#include <string>

namespace contra {

class ExprAST;
class PrototypeAST;
class JIT;
class DebugInfo;

class CodeGen {

  using AllocaInst = llvm::AllocaInst;
  using Function = llvm::Function;


public:

  JIT TheJIT;

  llvm::LLVMContext TheContext;
  llvm::IRBuilder<> Builder;
  std::unique_ptr<llvm::Module> TheModule;
  std::map<std::string, AllocaInst *> NamedValues;
  std::map<std::string, AllocaInst *> NamedArrays;
  std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM;
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
  
  // debug extras
  std::unique_ptr<llvm::DIBuilder> DBuilder;
  DebugInfo KSDbgInfo;

  // Constructor
  CodeGen (bool);

  Function *getFunction(std::string Name, int Line, int Depth=0); 

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  AllocaInst *createEntryBlockAlloca(Function *TheFunction,
    const std::string &VarName, VarTypes type, int Line, bool IsPointer=false);

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  std::pair<llvm::AllocaInst*, llvm::Value*> 
  createArray(Function *TheFunction,
    const std::string &VarName, VarTypes type, std::size_t NumVals, int Line,
    llvm::Value * SizeExpr = nullptr );

  void initArrays( Function *TheFunction, 
      const std::vector<AllocaInst*> & VarList,
      llvm::Value * InitVal,
      std::size_t NumVals,
      llvm::Value * SizeExpr = nullptr );

  void initArray( Function *TheFunction, 
      AllocaInst* Var,
      const std::vector<llvm::Value *> InitVals );
  
  void copyArrays( Function *TheFunction, 
      AllocaInst* Src,
      const std::vector<AllocaInst*> Tgts,
      llvm::Value * SizeExpr);

  void destroyArrays();


  /// Top-Level parsing and JIT Driver
  void initializeModuleAndPassManager();
  void initializeModule();
  void initializePassManager();

  // Return true if in debug mode
  auto isDebug() { return static_cast<bool>(DBuilder); }

  // JIT the current module
  JIT::VModuleKey doJIT();

  // Search the JIT for a symbol
  JIT::JITSymbol findSymbol( const char * Symbol );

  // Delete a JITed module
  void removeJIT( JIT::VModuleKey H );

  // Finalize whatever needs to be
  void finalize();

  llvm::DISubroutineType *createFunctionType(unsigned NumArgs, llvm::DIFile *Unit);


  // create a subprogram DIE
  llvm::DIFile * createFile();

  llvm::DISubprogram * createSubprogram(unsigned LineNo, unsigned ScopeLine,
      const std::string & Name, unsigned arg_size, llvm::DIFile *Unit);

  llvm::DILocalVariable *createVariable( llvm::DISubprogram *SP,
      const std::string & Name, unsigned ArgIdx, llvm::DIFile *Unit,
      unsigned LineNo, AllocaInst *Alloca);

  void emitLocation(ExprAST * ast) { 
    if (isDebug()) KSDbgInfo.emitLocation(Builder, ast);
  }
  
  void pushLexicalBlock(llvm::DISubprogram *SP) {
    if (isDebug()) KSDbgInfo.LexicalBlocks.push_back(SP);
  }
  
  void popLexicalBlock() {
    if (isDebug()) KSDbgInfo.LexicalBlocks.pop_back();
  }
};

} // namespace

#endif // CONTRA_CODEGEN_HPP
