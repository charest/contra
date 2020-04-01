#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "array.hpp"
#include "debug.hpp"
#include "dispatcher.hpp"
#include "tasking.hpp"
#include "jit.hpp"
#include "llvm_utils.hpp"
#include "scope.hpp"
#include "symbols.hpp" 

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

class CodeGen : public AstDispatcher, public Scoper {

  using AllocaInst = llvm::AllocaInst;
  using Function = llvm::Function;
  using Type = llvm::Type;
  using Value = llvm::Value;
  
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

  std::map<std::string, Type*> TypeTable_;
 
  class VariableEntry {
    Value* Alloca_ = nullptr;
    Type* Type_ = nullptr;
    Value* Size_ = nullptr;
    bool IsOwner_ = true;
  public:
    VariableEntry() = default;
    VariableEntry(Value* Alloca, Type* Type, Value* Size = nullptr)
      : Alloca_(Alloca), Type_(Type), Size_(Size) {}
    auto getAlloca() const { return Alloca_; }
    auto getType() const { return Type_; }
    auto getSize() const { return Size_; }
    void setOwner(bool IsOwner=true) { IsOwner_=IsOwner; }
    auto isOwner() const { return IsOwner_; }
  };

  using VariableTable = std::map<std::string, VariableEntry>;
  std::forward_list< VariableTable > VariableTable_;
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionTable_;

  // debug extras
  std::unique_ptr<llvm::DIBuilder> DBuilder;
  DebugInfo KSDbgInfo;

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
  {
    auto OrigScope = getScope();
    auto V = runExprVisitor(e);
    resetScope(OrigScope);
    return V;
  }

  // Visitees 
  void dispatch(ValueExprAST<int_t>&) override;
  void dispatch(ValueExprAST<real_t>&) override;
  void dispatch(ValueExprAST<std::string>&) override;
  void dispatch(VariableExprAST&) override;
  void dispatch(ArrayExprAST&) override;
  void dispatch(FutureExprAST&) override;
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
  void dispatch(TaskAST&) override;
  

  // visitor helper
  llvm::Value* codegenFunctionBody(FunctionAST& e);

  //============================================================================
  // Scope interface
  //============================================================================

  Scoper::value_type createScope() override {
    VariableTable_.push_front({});
    return Scoper::createScope();
  }
  
  void resetScope(Scoper::value_type Scope) override;
  
  //============================================================================
  // Type interface
  //============================================================================

  Type* getLLVMType(const VariableType & Ty)
  { return TypeTable_.at(Ty.getBaseType()->getName()); }
  
  Type* getLLVMType(const Identifier & Id)
  { return TypeTable_.at(Id.getName()); }


  template<typename T>
  Value* getTypeSize(Type* ElementType)
  { return ::getTypeSize<T>(Builder_, ElementType); }

  //============================================================================
  // Variable interface
  //============================================================================
  VariableEntry * createVariable(Function *TheFunction,
    const std::string &VarName, Type* VarType, bool IsGlobal=false);
  
  VariableEntry * getVariable(const std::string & VarName);

  VariableEntry * moveVariable(const std::string & From, const std::string & To);

  VariableEntry * insertVariable(const std::string &VarName, VariableEntry VarEntry);

  //============================================================================
  // Array interface
  //============================================================================
 
  // is the value an array
  bool isArray(Type* Ty);
  bool isArray(Value* Val);

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  VariableEntry * createArray(Function *TheFunction, const std::string &VarName,
      Type* PtrType, Value * SizeExpr );
  
  VariableEntry * createArray(Function *TheFunction, const std::string &VarName,
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
  
};

} // namespace

#endif // CONTRA_CODEGEN_HPP
