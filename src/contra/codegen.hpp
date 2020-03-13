#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "array.hpp"
#include "debug.hpp"
#include "dispatcher.hpp"
#include "jit.hpp"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"

#include <map>
#include <memory>
#include <string>

namespace contra {

class BinopPrecedence;
class PrototypeAST;
class JIT;
class DebugInfo;

class CodeGen : public AstDispatcher {

  using AllocaInst = llvm::AllocaInst;
  using Function = llvm::Function;

  llvm::LLVMContext TheContext_;
  llvm::IRBuilder<> Builder_;
  std::unique_ptr<llvm::Module> TheModule_;
  
  std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM_;
  JIT TheJIT_;

  llvm::Value* ValueResult_ = nullptr;
  llvm::Function* FunctionResult_ = nullptr;

  std::map<std::string, llvm::Type*> TypeTable_;

  std::map<std::string, AllocaInst *> NamedValues;
  std::map<std::string, AllocaInst *> NamedArrays;
  std::map<AllocaInst *, ArrayType> TempArrays;


  // debug extras
  std::unique_ptr<llvm::DIBuilder> DBuilder;
  DebugInfo KSDbgInfo;

public:
  
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
  

  // Constructor
  CodeGen (bool);

  // destructor
  virtual ~CodeGen() = default;

  // some accessors
  llvm::IRBuilder<> & getBuilder() { return Builder_; }
  llvm::LLVMContext & getContext() { return TheContext_; }
  llvm::Module & getModule() { return *TheModule_; }

  Function *getFunction(std::string Name); 

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  AllocaInst *createEntryBlockAlloca(Function *TheFunction,
    const std::string &VarName, llvm::Type* VarType);

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  ArrayType
  createArray(Function *TheFunction, const std::string &VarName,
      llvm::Type* PtrType, llvm::Value * SizeExpr );

  // Initializes a bunch of arrays with a value
  void initArrays( Function *TheFunction, 
      const std::vector<AllocaInst*> & VarList,
      llvm::Value * InitVal,
      llvm::Value * SizeExpr );

  // initializes an array with a list of values
  void initArray( Function *TheFunction, 
      AllocaInst* Var,
      const std::vector<llvm::Value *> InitVals );
  
  // copies one array to another
  void copyArrays( Function *TheFunction, 
      AllocaInst* Src,
      const std::vector<AllocaInst*> Tgts,
      llvm::Value * SizeExpr);

  // destroy all arrays
  void destroyArrays();

  /// Top-Level parsing and JIT Driver
  void initializeModuleAndPassManager();
  void initializeModule();
  void initializePassManager();

  void optimize(llvm::Function* F)
  { TheFPM_->run(*F); }

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

  void emitLocation(NodeAST * ast) { 
    if (isDebug()) KSDbgInfo.emitLocation(Builder_, ast);
  }
  
  void pushLexicalBlock(llvm::DISubprogram *SP) {
    if (isDebug()) KSDbgInfo.LexicalBlocks.push_back(SP);
  }
  
  void popLexicalBlock() {
    if (isDebug()) KSDbgInfo.LexicalBlocks.pop_back();
  }
 
  // Codegen function
  template<
    typename T,
    typename = typename std::enable_if_t<
      std::is_same<T, FunctionAST>::value || std::is_same<T, PrototypeAST>::value >
  >
  llvm::Function* runFuncVisitor(T&e)
  {
    FunctionResult_ = nullptr;
    e.accept(*this);
    return FunctionResult_;
  }

private:

  
  // Codegen function
  template<typename T>
  llvm::Value* runExprVisitor(T&e)
  {
    ValueResult_ = nullptr;
    e.accept(*this);
    return ValueResult_;
  }

  // Visitees 
  void dispatch(ValueExprAST<int_t>&) override;
  void dispatch(ValueExprAST<real_t>&) override;
  void dispatch(ValueExprAST<std::string>&) override;
  void dispatch(VariableExprAST&) override;
  void dispatch(ArrayExprAST&) override;
  void dispatch(CastExprAST&) override;
  void dispatch(UnaryExprAST&) override;
  void dispatch(BinaryExprAST&) override;
  void dispatch(CallExprAST&) override;
  void dispatch(ForStmtAST&) override;
  void dispatch(IfStmtAST&) override;
  void dispatch(VarDeclAST&) override;
  void dispatch(ArrayDeclAST&) override;
  void dispatch(PrototypeAST&) override;
  void dispatch(FunctionAST&) override;

};

} // namespace

#endif // CONTRA_CODEGEN_HPP
