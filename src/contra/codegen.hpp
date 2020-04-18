#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "debug.hpp"
#include "recursive.hpp"
#include "tasking.hpp"
#include "jit.hpp"
#include "symbols.hpp" 
#include "variable.hpp"

#include "utils/llvm_utils.hpp"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"

#include <forward_list>
#include <map>
#include <memory>
#include <string>

namespace contra {

class BinopPrecedence;
class PrototypeAST;
class JIT;
class DebugInfo;
 
class CodeGen : public RecursiveAstVisiter {

  using AllocaInst = llvm::AllocaInst;
  using Function = llvm::Function;
  using Type = llvm::Type;
  using Value = llvm::Value;
  using VariableTable = std::map<std::string, VariableAlloca>;
  
  // LLVM builder types  
  llvm::LLVMContext TheContext_;
  llvm::IRBuilder<> Builder_;
  std::unique_ptr<llvm::Module> TheModule_;
  std::unique_ptr<llvm::ExecutionEngine> TheEngine_;
  
  std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM_;
  JIT TheJIT_;

  // visitor results
  Value* ValueResult_ = nullptr;
  Function* FunctionResult_ = nullptr;

  // symbol tables
  std::map<std::string, Type*> TypeTable_;
  std::forward_list< VariableTable > VariableTable_;
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionTable_;

  // debug extras
  std::unique_ptr<llvm::DIBuilder> DBuilder;
  DebugInfo KSDbgInfo;

  // defined types
  Type* I64Type_ = nullptr;
  Type* F64Type_ = nullptr;
  Type* VoidType_ = nullptr;
  Type* ArrayType_ = nullptr;

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
      unsigned LineNo, Value *Alloca);

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
  
  // Codegen function
  template<typename T>
  Value* runStmtVisitor(T&e)
  { return runExprVisitor(e); }

  // Visitees 
  void visit(ValueExprAST&) override;
  void visit(VarAccessExprAST&) override;
  void visit(ArrayAccessExprAST&) override;
  void visit(ArrayExprAST&) override;
  void visit(CastExprAST&) override;
  void visit(UnaryExprAST&) override;
  void visit(BinaryExprAST&) override;
  void visit(CallExprAST&) override;
  void visit(ForStmtAST&) override;
  void visit(ForeachStmtAST&) override;
  void visit(IfStmtAST&) override;
  void visit(AssignStmtAST&) override;
  void visit(VarDeclAST&) override;
  void visit(PrototypeAST&) override;
  void visit(FunctionAST&) override;
  void visit(TaskAST&) override;
  void visit(IndexTaskAST&) override;
  

  // visitor helper
  llvm::Value* codegenFunctionBody(FunctionAST& e);

  //============================================================================
  // Scope interface
  //============================================================================

  void createScope() 
  {
    VariableTable_.push_front({});
  }
  
  void popScope();
 
  //============================================================================
  // Type interface
  //============================================================================

  Type* getLLVMType(const VariableType & Ty)
  { return TypeTable_.at(Ty.getBaseType()->getName()); }
  
  Type* getLLVMType(const Identifier & Id)
  { return TypeTable_.at(Id.getName()); }


  template<typename T>
  Value* getTypeSize(Type* ElementType)
  { return utils::getTypeSize<T>(Builder_, ElementType); }

  //============================================================================
  // Variable interface
  //============================================================================
  VariableAlloca * createVariable(Function *TheFunction,
    const std::string &VarName, Type* VarType, bool IsGlobal=false);
  
  VariableAlloca * getVariable(const std::string & VarName);

  void eraseVariable(const std::string &);
  VariableAlloca * moveVariable(const std::string & From, const std::string & To);

  VariableAlloca * insertVariable(const std::string &VarName, VariableAlloca VarEntry);
  VariableAlloca * insertVariable(const std::string &VarName, llvm::Value*, llvm::Type*);

  //============================================================================
  // Array interface
  //============================================================================
 
  // is the value an array
  bool isArray(Type* Ty);
  bool isArray(Value* Val);

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  VariableAlloca * createArray(Function *TheFunction, const std::string &VarName,
      Type* PtrType, Value * SizeExpr );
  
  VariableAlloca * createArray(Function *TheFunction, const std::string &VarName,
      Type* ElementType, bool IsGlobal=false);

  // Initializes a bunch of arrays with a value
  void initArrays( Function *TheFunction, 
      const std::vector<Value*> & VarList,
      Value * InitVal,
      Value * SizeExpr,
      Type * ElementType );

  // initializes an array with a list of values
  void initArray( Function *TheFunction, 
      Value* Var,
      const std::vector<Value *> InitVals,
      Type * ElementType );
  
  // copies one array to another
  void copyArrays( Function *TheFunction, 
      Value* Src,
      const std::vector<Value*> Tgts,
      Value * SizeExpr,
      Type * ElementType);

  void copyArray(Value* Src, Value* Tgt);

  // destroy all arrays
  void destroyArray(const std::string &, Value*);
  void destroyArrays(const std::map<std::string, Value*> &);

  // load an array value
  Value* loadArrayValue(Value*, Value*, Type*, const std::string &);

  // store an array value
  void storeArrayValue(Value*, Value*, Value*, const std::string &);

  // Load an array
  Value* loadArrayPointer(Value*, Type*, const std::string & = "");
  Value* createArrayPointerAlloca(Function *, Value*, Type*);
  std::vector<Value*> createArrayPointerAllocas(Function *, const std::vector<Value*> &, Type*);

  // get an arrays size
  Value* getArraySize(Value*, const std::string &);

  //============================================================================
  // Function interface
  //============================================================================

public: 
  PrototypeAST & insertFunction(std::unique_ptr<PrototypeAST> Proto);

private:
  Function *getFunction(std::string Name); 
  
  //============================================================================
  // Future interface
  //============================================================================
  llvm::Value* loadFuture(llvm::Type*, llvm::Value*);
  
};

} // namespace

#endif // CONTRA_CODEGEN_HPP
