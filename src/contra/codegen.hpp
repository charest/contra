#ifndef CONTRA_CODEGEN_HPP
#define CONTRA_CODEGEN_HPP

#include "recursive.hpp"
#include "tasking.hpp"
#include "jit.hpp"
#include "symbols.hpp" 
#include "variable.hpp"

#include "utils/llvm_utils.hpp"
#include "utils/builder.hpp"

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
 
class CodeGen : public RecursiveAstVisiter {

  using AllocaInst = llvm::AllocaInst;
  using Function = llvm::Function;
  using Type = llvm::Type;
  using Value = llvm::Value;
  using VariableTable = std::map<std::string, VariableAlloca>;
  
  // LLVM builder types  
  utils::Builder TheBuilder_;
  
  llvm::LLVMContext & TheContext_;
  llvm::IRBuilder<> & Builder_;
  std::unique_ptr<llvm::Module> TheModule_;
  std::unique_ptr<llvm::ExecutionEngine> TheEngine_;

  std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM_;
  JIT TheJIT_;

  // visitor results
  Value* ValueResult_ = nullptr;
  Function* FunctionResult_ = nullptr;
  Value* CurrentRange_ = nullptr;
  bool IsPacked_ = false;

  // symbol tables
  std::map<std::string, Type*> TypeTable_;
  std::forward_list< VariableTable > VariableTable_;
  std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionTable_;

  // defined types
  Type* I64Type_ = nullptr;
  Type* F64Type_ = nullptr;
  Type* VoidType_ = nullptr;
  Type* ArrayType_ = nullptr;
  Type* AccessorType_ = nullptr;

  // task interface
  std::unique_ptr<AbstractTasker> Tasker_;

  // command line arguments
  int Argc_ = 0;
  char ** Argv_ = nullptr;

  // temp counter
  std::size_t TmpCounter_ = 0;

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
  auto & getBuilder() { return Builder_; }
  auto & getContext() { return TheContext_; }
  auto & getModule() { return *TheModule_; }
  
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
  auto isDebug() { return false; }

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
    IsPacked_ = false;
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
  void visit(RangeExprAST&) override;
  void visit(CastExprAST&) override;
  void visit(UnaryExprAST&) override;
  void visit(BinaryExprAST&) override;
  void visit(CallExprAST&) override;
  void visit(ExprListAST&) override;

  void visit(ForStmtAST&) override;
  void visit(ForeachStmtAST&) override;
  void visit(IfStmtAST&) override;
  void visit(AssignStmtAST&) override;
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
  { VariableTable_.push_front({}); }
  
  void popScope();
 
  //============================================================================
  // Type interface
  //============================================================================

  Type* getLLVMType(const VariableType & Ty)
  {
    if (Ty.isArray()) return ArrayType_;
    if (Ty.isRange()) return Tasker_->getRangeType();
    if (Ty.isField()) { return Tasker_->getFieldType(); }
    if (Ty.isFuture()) { return Tasker_->getFutureType(); }
    if (Ty.isPartition()) return Tasker_->getPartitionType();
    if (Ty.isStruct()) {
      std::vector<llvm::Type*> Members;
      for (const auto & M : Ty.getMembers())
        Members.emplace_back( getLLVMType(M) );
      return llvm::StructType::create(Members, "struct.t");
    }
    return TypeTable_.at(Ty.getBaseType()->getName());
  }
  
  Type* getLLVMType(const Identifier & Id)
  { return TypeTable_.at(Id.getName()); }

  Type* getLLVMType(const std::string & Name)
  { return TypeTable_.at(Name); }

  bool isLLVMType(const std::string & Name)
  { return TypeTable_.count(Name); }
    

  std::string getTempName() {
    auto Name = "__tmp" + std::to_string(TmpCounter_);
    TmpCounter_++;
    return Name;
  }

  //============================================================================
  // Variable interface
  //============================================================================
  VariableAlloca * createVariable(
      Function *TheFunction,
      const std::string &VarName,
      Type* VarType);

  std::pair<VariableAlloca*, bool> getOrCreateVariable(
      const std::string &VarName,
      const VariableType &);
  
  VariableAlloca * getVariable(const std::string & VarName);

  VariableAlloca * insertVariable(
      const std::string &VarName,
      VariableAlloca VarEntry);
  
  VariableAlloca * insertVariable(
      const std::string &VarName,
      llvm::Value*,
      llvm::Type*);

  //============================================================================
  // Array interface
  //============================================================================
 
  // is the value an array
  bool isArray(Type* Ty);
  bool isArray(Value* Val);

  /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
  /// the function.  This is used for mutable variables etc.
  VariableAlloca * createArray(
      const std::string &VarName,
      Type* PtrType,
      Value * SizeExpr );
  
  VariableAlloca * createArray(
      const std::string &VarName,
      Type* ElementType);
  
  void createArray(
      Value* ArrayA,
      Value * SizeV,
      Type* ElementT);

  // Initializes a bunch of arrays with a value
  void initArrays(
      Function *TheFunction, 
      const std::vector<Value*> & VarList,
      Value * InitVal,
      Value * SizeExpr,
      Type * ElementType );

  // initializes an array with a list of values
  void initArray(
      Function *TheFunction, 
      Value* Var,
      const std::vector<Value *> InitVals,
      Type * ElementType );
  
  // copies one array to another
  void copyArrays(
      Function *TheFunction, 
      Value* Src,
      const std::vector<Value*> Tgts,
      Value * SizeExpr,
      Type * ElementType);

  void copyArray(Value* Src, Value* Tgt);

  // destroy all arrays
  void destroyArray(Value*);
  void destroyArrays(const std::vector<Value*> &);

  // load an array value
  Value* extractArrayValue(Value*, Type*, Value*);

  // store an array value
  void insertArrayValue(Value*, Type*, Value*, Value*);

  // Load an array
  Value* getArrayPointer(Value*, Type*);
  Value* getArrayElementPointer(Value*, Type*, Value*);
  
  Value* createArrayPointerAlloca(Value*, Type*);

  std::vector<Value*> createArrayPointerAllocas(
      const std::vector<Value*> &,
      Type*);

  // get an arrays size
  Value* getArraySize(Value*);
  
  //============================================================================
  // Range interface
  //============================================================================

  VariableAlloca * createRange(
      const std::string &VarName,
      Value* StartV,
      Value* EndV,
      Value* StepV,
      bool IsTask);

  //============================================================================
  // Function interface
  //============================================================================

public: 
  PrototypeAST & insertFunction(std::unique_ptr<PrototypeAST> Proto);

private:
  std::pair<Function*,bool> getFunction(std::string Name); 
  
  //============================================================================
  // Future interface
  //============================================================================
  llvm::Value* loadFuture(llvm::Type*, llvm::Value*);
  
  VariableAlloca * createField(
      const std::string &,
      Type*,
      Value*,
      Value* = nullptr);
};

} // namespace

#endif // CONTRA_CODEGEN_HPP
