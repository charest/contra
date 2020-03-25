#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "array.hpp"
#include "debug.hpp"
#include "dispatcher.hpp"
#include "tasking.hpp"
#include "jit.hpp"
#include "symbols.hpp" 

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
  using Type = llvm::Type;
  using Value = llvm::Value;

  // LLVM builder types  
  llvm::LLVMContext TheContext_;
  llvm::IRBuilder<> Builder_;
  std::unique_ptr<llvm::Module> TheModule_;
  
  std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM_;
  JIT TheJIT_;

  // visitor results
  Value* ValueResult_ = nullptr;
  Function* FunctionResult_ = nullptr;

  std::map<std::string, Type*> TypeTable_;
  std::map<std::string, AllocaInst*> VariableTable_;
  std::map<std::string, ArrayType> ArrayTable_;
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionTable_;
  std::map<std::string, TaskInfo> TaskTable_;

  // debug extras
  std::unique_ptr<llvm::DIBuilder> DBuilder;
  DebugInfo KSDbgInfo;

  Type* I64Type_ = nullptr;
  Type* F64Type_ = nullptr;
  Type* VoidType_ = nullptr;

  // task interface
  std::unique_ptr<AbstractTasker> Tasker_;

  // command line arguments
  int Argc_ = 0;
  char ** Argv_ = nullptr;

public:
  
  //============================================================================
  // Constructor / Destructors
  //============================================================================

  // Constructor
  CodeGen(bool);

  // destructor
  virtual ~CodeGen();

  //============================================================================
  // LLVM accessors
  //============================================================================

  // some accessors
  llvm::IRBuilder<> & getBuilder() { return Builder_; }
  llvm::LLVMContext & getContext() { return TheContext_; }
  llvm::Module & getModule() { return *TheModule_; }

  static llvm::IRBuilder<> createBuilder(Function *TheFunction);
  
  //============================================================================
  // Optimization / Module interface
  //============================================================================

  /// Top-Level parsing and JIT Driver
  void initializeModuleAndPassManager();
  void initializeModule();
  void initializePassManager();

  void optimize(Function* F)
  { TheFPM_->run(*F); }

  //============================================================================
  // JIT interface
  //============================================================================

  // JIT the current module
  JIT::VModuleKey doJIT();

  // Search the JIT for a symbol
  JIT::JITSymbol findSymbol( const char * Symbol );

  // Delete a JITed module
  void removeJIT( JIT::VModuleKey H );

  //============================================================================
  // Debug-related accessors
  //============================================================================
  
  // Return true if in debug mode
  auto isDebug() { return static_cast<bool>(DBuilder); }

  // create a debug entry for a function
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
  
  //============================================================================
  // Vizitor interface
  //============================================================================
 
  // Codegen function
  template<
    typename T,
    typename = typename std::enable_if_t<
      std::is_same<T, FunctionAST>::value || std::is_same<T, PrototypeAST>::value >
  >
  Function* runFuncVisitor(T&e)
  {
    FunctionResult_ = nullptr;
    e.accept(*this);
    return FunctionResult_;
  }

private:
  
  // Codegen function
  template<typename T>
  Value* runExprVisitor(T&e)
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
  
  //============================================================================
  // Type interface
  //============================================================================

  Type* getLLVMType(const VariableType & Ty)
  { return TypeTable_.at(Ty.getBaseType()->getName()); }
  
  Type* getLLVMType(const Identifier & Id)
  { return TypeTable_.at(Id.getName()); }

  //============================================================================
  // Variable interface
  //============================================================================
  AllocaInst *createVariable(Function *TheFunction,
    const std::string &VarName, Type* VarType);
  
  AllocaInst *getVariable(const std::string & VarName)
  { return VariableTable_.at(VarName); }

  AllocaInst* moveVariable(const std::string & From, const std::string & To);

  //============================================================================
  // Array interface
  //============================================================================

  auto getArray(const std::string & Name)
  { return ArrayTable_.at(Name); }

  ArrayType & moveArray(const std::string & From, const std::string & To);
  
  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  ArrayType
  createArray(Function *TheFunction, const std::string &VarName,
      Type* PtrType, Value * SizeExpr );

  // Initializes a bunch of arrays with a value
  void initArrays( Function *TheFunction, 
      const std::vector<AllocaInst*> & VarList,
      Value * InitVal,
      Value * SizeExpr );

  // initializes an array with a list of values
  void initArray( Function *TheFunction, 
      AllocaInst* Var,
      const std::vector<Value *> InitVals );
  
  // copies one array to another
  void copyArrays( Function *TheFunction, 
      AllocaInst* Src,
      const std::vector<AllocaInst*> Tgts,
      Value * SizeExpr);

  // destroy all arrays
  void destroyArrays();

  
  //============================================================================
  // Function interface
  //============================================================================

public: 
  PrototypeAST & insertFunction(std::unique_ptr<PrototypeAST> Proto);

private:
  Function *getFunction(std::string Name); 
  
  //============================================================================
  // Task interface
  //============================================================================

  TaskInfo & insertTask(const std::string & Name, Function* F);

  bool isTask(const std::string & Name) const
  { return TaskTable_.count(Name); }

  auto getTask(const std::string & Name) const
  { return TaskTable_.at(Name); }

public: 

  void updateTask(const std::string & Name);
  
};

} // namespace

#endif // CONTRA_CODEGEN_HPP
