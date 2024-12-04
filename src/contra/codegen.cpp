#include "config.hpp"

#include "ast.hpp"
#include "context.hpp"
#include "codegen.hpp"
#include "compiler.hpp"
#include "cuda.hpp"
#include "cuda_jit.hpp"
#include "errors.hpp"
#include "legion.hpp"
#include "mpi.hpp"
#include "precedence.hpp"
#include "rocm.hpp"
#include "rocm_jit.hpp"
#include "serial.hpp"
#include "threads.hpp"
#include "token.hpp"
#include "variable.hpp"

#include "librt/librt.hpp"
#include "librt/dopevector.hpp"

#include "utils/llvm_utils.hpp"

#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

using namespace llvm;
using namespace llvm::orc;
using namespace utils;

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Constructor / Destructor
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Constructor
//==============================================================================
CodeGen::CodeGen (
    SupportedBackends Backend,
    bool)
{
  HostJIT_ = ExitOnErr(JIT::Create());
  TheHelper_.setContext( HostJIT_->getContext() );

  // setup runtime
  librt::RunTimeLib::setup(getContext());
  
  // setup types
  I64Type_  = llvmType<int_t>(getContext());
  F64Type_  = llvmType<real_t>(getContext());
  VoidType_ = Type::getVoidTy(getContext());
  ArrayType_ = librt::DopeVector::DopeVectorType;
  BoolType_ = llvmType<bool>(getContext());
  
  // setup tasker
  Tasker_ = nullptr;

  if (Backend == SupportedBackends::Serial)
    Tasker_ = std::make_unique<SerialTasker>(TheHelper_);

#ifdef HAVE_THREADS
  else if (Backend == SupportedBackends::Threads) {
    Tasker_ = std::make_unique<ThreadsTasker>(TheHelper_);
  }
#endif

#ifdef HAVE_LEGION
  else if (Backend == SupportedBackends::Legion) {
    Tasker_ = std::make_unique<LegionTasker>(TheHelper_);
  }
#endif

#ifdef HAVE_CUDA
  else if (Backend == SupportedBackends::Cuda) {
    DeviceJIT_ = std::make_unique<CudaJIT>(TheHelper_);
    Tasker_ = std::make_unique<CudaTasker>(TheHelper_);
  }     
#endif
  
#ifdef HAVE_ROCM
  else if (Backend == SupportedBackends::ROCm) {
    Tasker_ = std::make_unique<ROCmTasker>(TheHelper_);
    DeviceJIT_ = std::make_unique<ROCmJIT>(TheHelper_);
  }     
#endif
  
#ifdef HAVE_MPI
  else if (Backend == SupportedBackends::MPI) {
    Tasker_ = std::make_unique<MpiTasker>(TheHelper_);
  }     
#endif
  
  else 
    THROW_CONTRA_ERROR("No viable backend selected!");

  Tasker_->registerSerializer<ArraySerializer>(
      ArrayType_,
      TheHelper_,
      librt::DopeVector::DopeVectorType);
  
  AccessorType_ = Tasker_->getAccessorType();

  // setup types
  auto & C = Context::instance();
  TypeTable_.emplace( C.getInt64Type()->getName(),  I64Type_);
  TypeTable_.emplace( C.getFloat64Type()->getName(),  F64Type_);
  TypeTable_.emplace( C.getVoidType()->getName(), VoidType_);
  TypeTable_.emplace( C.getBoolType()->getName(), BoolType_);

  VariableTable_.push_front({});

  // init function optimizer
  initializeModuleAndPassManager();

}
  
////////////////////////////////////////////////////////////////////////////////
// Optimization / Module interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Initialize module and optimizer
//==============================================================================
void CodeGen::initializeModuleAndPassManager()
{
  initializeModule();
  if (!isDebug())
    initializePassManager();
}

//==============================================================================
// Initialize module
//==============================================================================
void CodeGen::initializeModule()
{
  // Open a new module.
  TheModule_ = std::make_unique<Module>("my cool jit", getContext());
  TheModule_->setDataLayout(HostJIT_->getDataLayout());
}

//==============================================================================
// Initialize optimizer
//==============================================================================
void CodeGen::initializePassManager() {

  // Create a new pass manager attached to it.
  TheFPM_ = std::make_unique<legacy::FunctionPassManager>(TheModule_.get());

  // Promote allocas to registers.
  TheFPM_->add(createPromoteMemoryToRegisterPass());
  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM_->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM_->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM_->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM_->add(createCFGSimplificationPass());
  TheFPM_->doInitialization();
}

////////////////////////////////////////////////////////////////////////////////
// JIT interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// JIT the current module
//==============================================================================
JIT::Resource CodeGen::doJIT(bool newResource)
{
  //std::cout << "----->>>>>>> DOJIT" << std::endl;
  //TheModule_->print(outs(), nullptr); outs()<<"\n";
  JIT::Resource RT = newResource ? HostJIT_->createResource() : nullptr;
  auto H = HostJIT_->addModule(std::move(TheModule_), RT);
  initializeModuleAndPassManager();
  return H;
}

//==============================================================================
// Search the JIT for a symbol
//==============================================================================
Expected<ExecutorSymbolDef> CodeGen::findSymbol( const char * Symbol )
{ return HostJIT_->findSymbol(Symbol); }

//==============================================================================
// Delete a JITed module
//==============================================================================
void CodeGen::removeJIT( JIT::Resource H )
{ HostJIT_->removeModule(H); }


////////////////////////////////////////////////////////////////////////////////
// Scope Interface
////////////////////////////////////////////////////////////////////////////////  
void CodeGen::popScope()
{
  std::vector<Value*> Arrays;
  std::vector<Value*> Futures;
  std::vector<Value*> Fields;
  std::vector<Value*> Ranges;
  std::vector<Value*> Accessors;
  std::vector<Value*> Partitions;

  for ( const auto & entry_pair : VariableTable_.front() ) {
    auto VarE = entry_pair.second;
    destroyVariable(VarE);
  }
  VariableTable_.pop_front();

}


////////////////////////////////////////////////////////////////////////////////
// Type Interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Convert our types to llvm types
//==============================================================================
Type* CodeGen::getLLVMType(const VariableType & Ty)
{
  if (Ty.isStruct()) {
    std::vector<llvm::Type*> Members;
    for (const auto & M : Ty.getMembers())
      Members.emplace_back( getLLVMType(M) );
     auto res = StructTable_.emplace(
         Members, 
         llvm::StructType::create(Members, "struct.t"));
    return res.first->second;
  }
  auto VarT = TypeTable_.at(Ty.getBaseType()->getName());
  if (Ty.isArray()) return ArrayType_;
  if (Ty.isRange()) return Tasker_->getRangeType(VarT);
  if (Ty.isField()) return Tasker_->getFieldType(VarT);
  if (Ty.isFuture()) return Tasker_->getFutureType(VarT);
  if (Ty.isPartition()) return Tasker_->getPartitionType(VarT);
  return VarT; 
}

////////////////////////////////////////////////////////////////////////////////
// Variable Interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Get a variable
//==============================================================================
VariableAlloca * CodeGen::getVariable(const std::string & VarName)
{ 
  for ( auto & ST : VariableTable_ ) {
    auto it = ST.find(VarName);
    if (it != ST.end()) return &it->second;
  }
  return nullptr;
}

//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
VariableAlloca * CodeGen::createVariable(
    StringRef VarName,
    Type* VarType)
{
  auto NewVar = TheHelper_.createEntryBlockAlloca(VarType, VarName);
  return insertVariable(VarName, VariableAlloca{NewVar, VarType});
}
        
//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
std::pair<VariableAlloca*, bool> CodeGen::getOrCreateVariable(
    const std::string &VarName,
    const VariableType & VarType)
{
  for ( auto & ST : VariableTable_ ) {
    auto it = ST.find(VarName);
    if (it != ST.end()) return {&it->second, false};
  }

  AllocaInst* NewVar = nullptr;
  Type* VarT = nullptr;

  if (VarType.isArray()) {
    VarT = getLLVMType(VarType.getIndexedType());
    NewVar = TheHelper_.createEntryBlockAlloca(ArrayType_, VarName);
  }
  else {
    VarT = getLLVMType(VarType);
    NewVar = TheHelper_.createEntryBlockAlloca(VarT, VarName);
  }
  
  auto VarE = insertVariable(VarName, VariableAlloca{NewVar, VarT});
  
  return {VarE, true};
}
        
//==============================================================================
/// Insert an already allocated variable
//==============================================================================
VariableAlloca * CodeGen::insertVariable(
    StringRef VarName,
    const VariableAlloca & VarE)
{ 
  auto it = VariableTable_.front().emplace(VarName, VarE);
  return &it.first->second;
}

//==============================================================================
/// Insert an already allocated variable
//==============================================================================
VariableAlloca * CodeGen::insertVariable(
    StringRef VarName,
    AllocaInst* VarAlloca,
    Type* VarType)
{ 
  VariableAlloca VarE(VarAlloca, VarType);
  return insertVariable(VarName, VarE);
}

//==============================================================================
// Destroy a variable
//==============================================================================
void CodeGen::destroyVariable(const VariableAlloca & VarE) {
  if (!VarE.isOwner()) return;

  auto Alloca = VarE.getAlloca();
  if (isArray(Alloca))
    destroyArray(Alloca); 
  else if (Tasker_->isFuture(Alloca))
    Tasker_->destroyFuture(*TheModule_, Alloca);
  else if (Tasker_->isField(Alloca))
    Tasker_->destroyField(*TheModule_, Alloca);
  else if (Tasker_->isRange(Alloca))
    Tasker_->destroyRange(*TheModule_, Alloca);
  else if (Tasker_->isAccessor(Alloca))
    Tasker_->destroyAccessor(*TheModule_, Alloca);
  // Index partitions are deleted with index spaces
  //else if (Tasker_->isPartition(Alloca))
  //  Tasker_->destroyPartition(*TheModule_, Alloca);
}

//==============================================================================
// Delete a variable
//==============================================================================
void CodeGen::eraseVariable(const std::string & VarN )
{
  for ( auto & ST : VariableTable_ ) {
    auto it = ST.find(VarN);
    if (it != ST.end()) ST.erase(it);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Array Interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
///  Is the variable an array
//==============================================================================
bool CodeGen::isArray(Type* Ty)
{ return librt::DopeVector::isDopeVector(Ty); }

bool CodeGen::isArray(Value* V)
{ return librt::DopeVector::isDopeVector(V); }

//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
VariableAlloca * CodeGen::createArray(
    llvm::StringRef VarName,
    Type* ElementT)
{
  auto NewVar = TheHelper_.createEntryBlockAlloca(ArrayType_, VarName);
  return insertVariable(VarName, NewVar, ElementT);
}


//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
VariableAlloca * CodeGen::createArray(
    llvm::StringRef VarName,
    Type * ElementType,
    Value * SizeExpr)
{

  SizeExpr = TheHelper_.getAsValue(SizeExpr);
  auto ArrayA = TheHelper_.createEntryBlockAlloca(ArrayType_, VarName+"vec");

  allocateArray(ArrayA, SizeExpr, ElementType);
  
  return insertVariable(VarName, {ArrayA, ElementType, SizeExpr});
}

//==============================================================================
// Allocate array
//==============================================================================
void CodeGen::allocateArray(
    Value* ArrayA,
    Value * SizeV,
    Type * ElementT)
{
  Function *F; 
  const auto & AllocateN = librt::DopeVectorAllocate::Name;
  F = TheModule_->getFunction(AllocateN);
  if (!F) F = librt::RunTimeLib::tryInstall(getContext(), *TheModule_, AllocateN);


  auto DataSize = TheHelper_.getTypeSize<int_t>(ElementT);
  std::vector<Value*> ArgVs = {SizeV, DataSize, ArrayA};
  getBuilder().CreateCall(F, ArgVs);
}
 
 
//==============================================================================
// Initialize Array
//==============================================================================
void CodeGen::initArray(
    Value* ArrayA,
    Value * InitV,
    Value * SizeV,
    Type * ElementT)
{
  auto TheFunction = getBuilder().GetInsertBlock()->getParent();
  
  // create allocas to the pointers
  auto ArrayPtrA = createArrayPointerAlloca(ArrayA, ElementT);

  // create a loop
  auto Alloca = TheHelper_.createEntryBlockAlloca(I64Type_, "__i");
  Value * StartVal = llvmValue<int_t>(getContext(), 0);
  getBuilder().CreateStore(StartVal, Alloca);
  
  auto BeforeBB = BasicBlock::Create(getContext(), "beforeinit", TheFunction);
  auto LoopBB =   BasicBlock::Create(getContext(), "init", TheFunction);
  auto AfterBB =  BasicBlock::Create(getContext(), "afterinit", TheFunction);
  getBuilder().CreateBr(BeforeBB);
  getBuilder().SetInsertPoint(BeforeBB);
  auto CurVar = getBuilder().CreateLoad(I64Type_, Alloca);
  SizeV = TheHelper_.getAsValue(SizeV);
  auto EndCond = getBuilder().CreateICmpSLT(CurVar, SizeV, "initcond");
  getBuilder().CreateCondBr(EndCond, LoopBB, AfterBB);
  getBuilder().SetInsertPoint(LoopBB);

  // store value
  InitV = TheHelper_.getAsValue(InitV);
  insertArrayValue(ArrayPtrA, ElementT, CurVar, InitV);

  // increment loop
  auto StepVal = llvmValue<int_t>(getContext(), 1);
  TheHelper_.increment(Alloca, StepVal);
  getBuilder().CreateBr(BeforeBB);
  getBuilder().SetInsertPoint(AfterBB);
}
  
//==============================================================================
// Initialize an arrays individual values
//==============================================================================
void CodeGen::initArray(
    Value* ArrayA,
    const std::vector<Value *> InitVs,
    Type* ElementT )
{
  auto NumVals = InitVs.size();
  
  // create allocas to the pointers
  auto ArrayPtrA = createArrayPointerAlloca(ArrayA, ElementT);
  for (std::size_t i=0; i<NumVals; ++i) {
    auto IndexV = llvmValue<int_t>(getContext(), i);
    insertArrayValue(ArrayPtrA, ElementT, IndexV, InitVs[i]);
  }
}

//==============================================================================
// Copy Array
//==============================================================================
void CodeGen::copyArray(Value* SrcArrayV, Value* TgtArrayA)
{
  const auto & CopyN = librt::DopeVectorCopy::Name;
  auto F = TheModule_->getFunction(CopyN);
  if (!F) F = librt::RunTimeLib::tryInstall(getContext(), *TheModule_, CopyN);

  auto SrcArrayA = TheHelper_.getAsAlloca(SrcArrayV);

  std::vector<Value*> Args = {SrcArrayA, TgtArrayA};
  getBuilder().CreateCall(F, Args);
}

//==============================================================================
// Load an arrayarray into an alloca
//==============================================================================
AllocaInst* CodeGen::createArrayPointerAlloca(
    Value* ArrayA,
    Type* ElementT )
{
  auto ElementPtrT = ElementT->getPointerTo();
  auto ArrayV = getArrayPointer(ArrayA, ElementT);
  auto ArrayPtrA = TheHelper_.createEntryBlockAlloca(ElementPtrT);
  getBuilder().CreateStore(ArrayV, ArrayPtrA);
  return ArrayPtrA;
}

//==============================================================================
/// Load an array pointer from the dopevecter
//==============================================================================
Value* CodeGen::getArrayPointer(
    Value* ArrayA,
    Type * ElementT)
{
  auto PtrV = TheHelper_.extractValue(ArrayA, 0);
  auto PtrT = ElementT->getPointerTo();
  Value* CastV = TheHelper_.createBitCast(PtrV, PtrT);
  return CastV;
}

//==============================================================================
// Get a pointer to an element in an array
//==============================================================================
Value* CodeGen::getArrayElementPointer(
    Value* ArrayA,
    Type* ElementT,
    Value* IndexV)
{
  auto ArrayPtrV = getArrayPointer(ArrayA, ElementT);
  IndexV = TheHelper_.getAsValue(IndexV);
  return getBuilder().CreateGEP(ElementT, ArrayPtrV, IndexV);
}

//==============================================================================
// Load an array value
//==============================================================================
Value* CodeGen::getArraySize(Value* ArrayA)
{ return TheHelper_.extractValue(ArrayA, 1); }
  

//==============================================================================
// Load an array value
//==============================================================================
Value* CodeGen::extractArrayValue(
    Value* ArrayA,
    Type* ElementT,
    Value* IndexV)
{
  Value* ArrayGEP = nullptr;
  if (isArray(ArrayA))
    ArrayGEP = getArrayElementPointer(ArrayA, ElementT, IndexV);
  else
    ArrayGEP = TheHelper_.offsetPointer(ElementT, ArrayA, IndexV);
  return getBuilder().CreateLoad(ElementT, ArrayGEP);
}

//==============================================================================
// Store array value
//==============================================================================
void CodeGen::insertArrayValue(
    Value* ArrayA,
    Type* ElementT,
    Value* IndexV,
    Value* ValueV)
{
  ValueV = TheHelper_.getAsValue(ValueV);
  Value* ArrayPtr = nullptr;
  if (isArray(ArrayA))
    ArrayPtr = getArrayElementPointer(ArrayA, ElementT, IndexV);
  else
    ArrayPtr = TheHelper_.offsetPointer(ElementT, ArrayA, IndexV);
  getBuilder().CreateStore(ValueV, ArrayPtr);
}
  
  
//==============================================================================
// Destroy an array
//==============================================================================
void CodeGen::destroyArray(Value* Alloca)
{
  const auto & DeallocateN = librt::DopeVectorDeAllocate::Name;
  auto F = TheModule_->getFunction(DeallocateN);
  if (!F) F = librt::RunTimeLib::tryInstall(getContext(), *TheModule_, DeallocateN);

  getBuilder().CreateCall(F, Alloca);
}

//==============================================================================
// Destroy all arrays
//==============================================================================
void CodeGen::destroyArrays(const std::vector<Value*> & Arrays)
{
  if (Arrays.empty()) return;

  for ( auto Array : Arrays ) destroyArray(Array);
}


////////////////////////////////////////////////////////////////////////////////
// Function Interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Get the function
//==============================================================================
std::pair<Function*, bool> CodeGen::getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto F = TheModule_->getFunction(Name))
    return {F,false};
  
  // see if this is an available intrinsic, try installing it first
  if (auto F = librt::RunTimeLib::tryInstall(getContext(), *TheModule_, Name))
    return {F,false};

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto fit = FunctionTable_.find(Name);
  return {runFuncVisitor(*fit->second), fit->second->getReturnType().isStruct()};
}

//==============================================================================
// Insert the function
//==============================================================================
PrototypeAST & CodeGen::insertFunction(std::unique_ptr<PrototypeAST> Proto)
{
  auto & P = FunctionTable_[Proto->getName()];
  P = std::move(Proto);
  return *P;
}

////////////////////////////////////////////////////////////////////////////////
// Visitor Interface
////////////////////////////////////////////////////////////////////////////////


//==============================================================================
// IntegerExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
void CodeGen::visit(ValueExprAST & e)
{
  switch (e.getValueType()) {
  case ValueExprAST::ValueType::Int:
    ValueResult_ = llvmValue<int_t>(getContext(), e.getVal<int_t>());
    break;
  case ValueExprAST::ValueType::Real:
    ValueResult_ = llvmValue(getContext(), e.getVal<real_t>());
    break;
  case ValueExprAST::ValueType::String:
    ValueResult_ = llvmString(getContext(), *TheModule_, e.getVal<std::string>());
    break;
  }
}
 
//==============================================================================
// VarAccessExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
void CodeGen::visit(VarAccessExprAST& e)
{

  auto Name = e.getName();

  // Look this variable up in the function.
  auto VarE = getVariable(Name);
  Value * VarA = VarE->getAlloca();
  
  // Load the value.
  if (Tasker_->isAccessor(VarA)) {
    ValueResult_ = Tasker_->loadAccessor(*TheModule_, VarE->getType(), VarA);
  }
  else {
    ValueResult_ = VarA;
  }
}

//==============================================================================
// VarAccessExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
void CodeGen::visit(ArrayAccessExprAST& e)
{

  const auto & Name = e.getName();

  // Look this variable up in the function.
  auto VarE = getVariable(Name);
  auto VarA = VarE->getAlloca();
  
  // Load the value.
  auto IndexV = runExprVisitor(*e.getIndexExpr());

  if (Tasker_->isAccessor(VarA)) {
    ValueResult_ = Tasker_->loadAccessor(*TheModule_, VarE->getType(), VarA, IndexV);
  }
  else if (Tasker_->isRange(VarA)) {
    ValueResult_ = Tasker_->loadRangeValue(VarA, IndexV);
  }
  else {
    ValueResult_ = extractArrayValue(VarA, VarE->getType(), IndexV);
  }
}


//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
void CodeGen::visit(ArrayExprAST &e)
{
  // the llvm variable type
  auto VarType = setArray(e.getType(), false);
  auto VarT = getLLVMType(VarType);

  std::vector<Value*> InitVals;
  InitVals.reserve(e.getNumVals());
  for ( const auto & E : e.getValExprs() ) 
    InitVals.emplace_back( runExprVisitor(*E) );

  Value* SizeExpr = nullptr;
  if (e.hasSize()) {
    SizeExpr = runExprVisitor(*e.getSizeExpr());
  }
  else {
    SizeExpr = llvmValue<int_t>(getContext(), e.getNumVals());
  }

  auto ArrayN = e.getName();
  auto ArrayE = createArray(ArrayN, VarT, SizeExpr );
  auto ArrayA = ArrayE->getAlloca();

  if (e.hasSize()) 
    initArray(ArrayA, InitVals[0], SizeExpr, VarT);
  else
    initArray(ArrayA, InitVals, VarT);

  ValueResult_ = ArrayA;
}

//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
void CodeGen::visit(RangeExprAST &e)
{
  // the llvm variable type
  auto StartV = runExprVisitor(*e.getStartExpr());
  auto EndV = runExprVisitor(*e.getEndExpr());
  Value* StepV = nullptr;
  if (e.hasStepExpr()) StepV = runExprVisitor(*e.getStepExpr());


  auto RangeN = e.getName();

  auto RangeA = Tasker_->createRange(
      *TheModule_,
      RangeN,
      StartV,
      EndV,
      StepV);

  auto RangeT = RangeA->getAllocatedType();
  insertVariable(RangeN, RangeA, RangeT);

  ValueResult_ = RangeA;
}
  
//==============================================================================
// CastExprAST - Expression class for casts.
//==============================================================================
void CodeGen::visit(CastExprAST &e)
{
  auto ToType = e.getType();
  auto ToT = getLLVMType(ToType);
  auto FromV = runExprVisitor(*e.getFromExpr());

  if (ToType.isStruct()) {
    auto ToA = TheHelper_.createEntryBlockAlloca(ToT);

    const auto & ToMembers = ToType.getMembers();
    auto NumElem = ToMembers.size();
  
    for (unsigned i=0; i<NumElem; ++i) {
      auto StructT = cast<StructType>(ToT);
      auto MemberT = StructT->getElementType(i);
      Value* MemberV = TheHelper_.extractValue(FromV, i);
      MemberV = TheHelper_.createCast(MemberV, MemberT);
      TheHelper_.insertValue(ToA, MemberV, i); 
    }
    ValueResult_ = ToA;
  }
  else {
    FromV = TheHelper_.getAsValue(FromV);
    ValueResult_ = TheHelper_.createCast(FromV, ToT);
  }

}

//==============================================================================
// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
void CodeGen::visit(UnaryExprAST & e) {
  auto OperandV = TheHelper_.getAsValue( runExprVisitor(*e.getOpExpr()) );
  
  if (OperandV->getType()->isFloatingPointTy()) {
  
    switch (e.getOperand()) {
    case tok_sub:
      ValueResult_ = getBuilder().CreateFNeg(OperandV, "negtmp");
      return;
    }

  }
  else {
    switch (e.getOperand()) {
    case tok_sub:
      ValueResult_ = getBuilder().CreateNeg(OperandV, "negtmp");
      return;
    }
  }

  auto F = getFunction(std::string("unary") + e.getOperand()).first;
  ValueResult_ = getBuilder().CreateCall(F, OperandV, "unop");
}

//==============================================================================
// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
void CodeGen::visit(BinaryExprAST& e) {
  
  Value *L = TheHelper_.getAsValue( runExprVisitor(*e.getLeftExpr()) );
  Value *R = TheHelper_.getAsValue( runExprVisitor(*e.getRightExpr()) );

  auto l_is_real = L->getType()->isFloatingPointTy();
  auto r_is_real = R->getType()->isFloatingPointTy();
  bool is_real =  (l_is_real && r_is_real);

  if (is_real) {
    switch (e.getOperand()) {
    case tok_add:
      ValueResult_ = getBuilder().CreateFAdd(L, R, "addtmp");
      return;
    case tok_sub:
      ValueResult_ = getBuilder().CreateFSub(L, R, "subtmp");
      return;
    case tok_mul:
      ValueResult_ = getBuilder().CreateFMul(L, R, "multmp");
      return;
    case tok_div:
      ValueResult_ = getBuilder().CreateFDiv(L, R, "divtmp");
      return;
    case tok_mod:
      ValueResult_ = getBuilder().CreateFRem(L, R, "remtmp");
      return;
    case tok_lt:
      ValueResult_ = getBuilder().CreateFCmpULT(L, R, "cmptmp");
      return;
    case tok_le:
      ValueResult_ = getBuilder().CreateFCmpULE(L, R, "cmptmp");
      return;
    case tok_gt:
      ValueResult_ = getBuilder().CreateFCmpUGT(L, R, "cmptmp");
      return;
    case tok_ge:
      ValueResult_ = getBuilder().CreateFCmpUGE(L, R, "cmptmp");
      return;
    case tok_eq:
      ValueResult_ = getBuilder().CreateFCmpUEQ(L, R, "cmptmp");
      return;
    case tok_ne:
      ValueResult_ = getBuilder().CreateFCmpUNE(L, R, "cmptmp");
      return;
    } 
  }
  else {
    switch (e.getOperand()) {
    case tok_add:
      ValueResult_ = getBuilder().CreateAdd(L, R, "addtmp");
      return;
    case tok_sub:
      ValueResult_ = getBuilder().CreateSub(L, R, "subtmp");
      return;
    case tok_mul:
      ValueResult_ = getBuilder().CreateMul(L, R, "multmp");
      return;
    case tok_div:
      ValueResult_ = getBuilder().CreateSDiv(L, R, "divtmp");
      return;
    case tok_mod:
      ValueResult_ = getBuilder().CreateSRem(L, R, "divtmp");
      return;
    case tok_lt:
      ValueResult_ = getBuilder().CreateICmpSLT(L, R, "cmptmp");
      return;
    case tok_le:
      ValueResult_ = getBuilder().CreateICmpSLE(L, R, "cmptmp");
      return;
    case tok_gt:
      ValueResult_ = getBuilder().CreateICmpSGT(L, R, "cmptmp");
      return;
    case tok_ge:
      ValueResult_ = getBuilder().CreateICmpSGE(L, R, "cmptmp");
      return;
    case tok_eq:
      ValueResult_ = getBuilder().CreateICmpEQ(L, R, "cmptmp");
      return;
    case tok_ne:
      ValueResult_ = getBuilder().CreateICmpNE(L, R, "cmptmp");
      return;
    }
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  auto F = getFunction(std::string("binary") + e.getOperand()).first;

  Value *Ops[] = { L, R };
  ValueResult_ = getBuilder().CreateCall(F, Ops, "binop");
}

//==============================================================================
// CallExprAST - Expression class for function calls.
//==============================================================================
void CodeGen::visit(CallExprAST &e) {

  // Look up the name in the global module table.
  auto Name = e.getName();
  auto NumArgs = e.getNumArgs();

  // helpers to get arguments
  auto getArgValue = [&](auto i) { 
    auto Arg = runExprVisitor(*e.getArgExpr(i));
    return TheHelper_.getAsValue(Arg);
  };
  auto getArg = [&](auto i)
  { return runExprVisitor(*e.getArgExpr(i)); };

  // check if its a cast
  if (isLLVMType(Name)) {
    auto ToT = getLLVMType(Name);
    ValueResult_ = TheHelper_.createCast(getArgValue(0), ToT);
    return;
  }
  else if (Name == "len") {
    ValueResult_ = nullptr;
    auto Arg = getArg(0);
    if (Tasker_->isRange(Arg))
      ValueResult_ = Tasker_->getRangeSize(Arg);
    return;
  }
  else if (Name == "part") {
    if (NumArgs == 2) {
      ValueResult_ = Tasker_->createPartition(
          *TheModule_,
          getArg(0),
          getArg(1));
    }
    else if (NumArgs == 3) {
      ValueResult_ = Tasker_->createPartition(
        *TheModule_,
        getArg(0),
        getArg(1),
        getArg(2));
    }
    return;
  }

  auto FunPair = getFunction(Name);
  auto CalleeF = FunPair.first;
  auto IsTask = Tasker_->isTask(Name);
    
  std::vector<Value *> ArgVs;
  for (unsigned i = 0; i<e.getNumArgs(); ++i) {
    Value* Arg = getArg(i);
    if (!IsTask && Tasker_->isFuture(Arg)) {
      auto ArgT = getLLVMType( e.getArgType(i) );
      Arg = Tasker_->loadFuture(*TheModule_, Arg, ArgT);
    }
    ArgVs.push_back(Arg);
  }

  //----------------------------------------------------------------------------
  if (IsTask) {
    const auto & TaskI = Tasker_->getTask(Name);
    Value* FutureV = nullptr;

    if (e.isTopLevelTask()) {
      if (Tasker_->isStarted())
        Tasker_->postregisterTasks(*TheModule_);
      else
        Tasker_->preregisterTasks(*TheModule_);
      Tasker_->setTopLevelTask(*TheModule_, TaskI);
      Tasker_->start(*TheModule_);
    }
    else {
      FutureV = Tasker_->launch(*TheModule_, TaskI, ArgVs);
    }
  
    ValueResult_ = UndefValue::get(Type::getVoidTy(getContext()));

    auto CalleeT = CalleeF->getFunctionType()->getReturnType();
    if (!CalleeT->isVoidTy()) ValueResult_ = FutureV;
    if (!CalleeT->isVoidTy() && FutureV) {
      auto Ty = e.getType();
      if (!Ty.isFuture()) {
        auto ResultT = getLLVMType(Ty);
        FutureV = Tasker_->loadFuture(*TheModule_, FutureV, ResultT);
      }
      ValueResult_ = FutureV;
    }

  }
  //----------------------------------------------------------------------------
  else {
    std::string TmpN = CalleeF->getReturnType()->isVoidTy() ? "" : "calltmp";

    if (Name == "print") Tasker_->pushRootGuard(*TheModule_);

    for (auto & A : ArgVs) A = TheHelper_.getAsValue(A);
    ValueResult_ = getBuilder().CreateCall(CalleeF, ArgVs, TmpN);

    if (Name == "print") {
      Tasker_->popRootGuard(*TheModule_);
      ValueResult_ = UndefValue::get(Type::getVoidTy(getContext()));
    }
  }

  IsPacked_ = FunPair.second;

}

//==============================================================================
// expression list
//==============================================================================
void CodeGen::visit(ExprListAST & e) {

  auto StructT = getLLVMType(e.getType());
  auto StructA = TheHelper_.createEntryBlockAlloca(StructT, "struct.a");
    
  unsigned i = 0;
  for (const auto & Expr : e.getExprs()) {
    auto MemberV = runExprVisitor(*Expr);
    TheHelper_.insertValue(StructA, MemberV, i);
    i++;
  }

  ValueResult_ = StructA;
}

//==============================================================================
// Break statement
//==============================================================================
void CodeGen::visit(BreakStmtAST &) {
  if (ExitBlock_) getBuilder().CreateBr(ExitBlock_);
}


//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
void CodeGen::visit(IfStmtAST & e) {

  if ( e.getThenExprs().empty() && e.getElseExprs().empty() ) {
    ValueResult_ = Constant::getNullValue(VoidType_);
    return;
  }

  Value *CondV = runExprVisitor(*e.getCondExpr());
  CondV = TheHelper_.getAsValue(CondV);
  auto CondT = CondV->getType();
  if (CondT->isIntegerTy()) {
    auto ZeroC = llvmValue(getContext(), CondT, 0);
    CondV = getBuilder().CreateICmpNE(CondV, ZeroC, "cmptmp");
  }


  auto TheFunction = getBuilder().GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(getContext(), "then", TheFunction);
  BasicBlock *ElseBB = e.getElseExprs().empty() ? nullptr : BasicBlock::Create(getContext(), "else");
  BasicBlock *MergeBB = BasicBlock::Create(getContext(), "ifcont");

  if (ElseBB)
    getBuilder().CreateCondBr(CondV, ThenBB, ElseBB);
  else
    getBuilder().CreateCondBr(CondV, ThenBB, MergeBB);

  // Emit then value.
  getBuilder().SetInsertPoint(ThenBB);

  createScope();
  for ( const auto & stmt : e.getThenExprs() ) runStmtVisitor(*stmt);
  popScope();

  // get first non phi instruction
  ThenBB->getFirstNonPHI();

  getBuilder().CreateBr(MergeBB);

  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = getBuilder().GetInsertBlock();

  if (ElseBB) {

    // Emit else block.
    TheFunction->insert(TheFunction->end(), ElseBB);
    getBuilder().SetInsertPoint(ElseBB);

    createScope();
    for ( const auto & stmt : e.getElseExprs() ) runStmtVisitor(*stmt); 
    popScope();

    // get first non phi
    ElseBB->getFirstNonPHI();

    getBuilder().CreateBr(MergeBB);
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = getBuilder().GetInsertBlock();

  } // else

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  getBuilder().SetInsertPoint(MergeBB);
  //PHINode *PN = Builder.CreatePHI(ThenV->getType(), 2, "iftmp");

  //if (ThenV) PN->addIncoming(ThenV, ThenBB);
  //if (ElseV) PN->addIncoming(ElseV, ElseBB);
  //return PN;
  
  // for expr always returns 0.
  ValueResult_ = UndefValue::get(VoidType_);
}

//==============================================================================
// ForExprAST - Expression class for for/in.
//
// Output for-loop as:
//   ...
//   start = startexpr
//   goto loop
// loop:
//   variable = phi [start, loopheader], [nextvariable, loopend]
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   nextvariable = variable + step
//   endcond = endexpr
//   br endcond, loop, endloop
// outloop:
//==============================================================================
void CodeGen::visit(ForStmtAST& e) {
  
  auto TheFunction = getBuilder().GetInsertBlock()->getParent();
  
  createScope();

  // Create an alloca for the variable in the entry block.
  const auto & VarN = e.getVarName();
  auto VarT = I64Type_;
  auto VarE = createVariable(VarN, VarT);
  auto VarA = VarE->getAlloca();
  
  // Emit the start code first, without 'variable' in scope.
  auto EndA = TheHelper_.createEntryBlockAlloca(VarT, VarN+"end");
  auto StepA = TheHelper_.createEntryBlockAlloca(VarT, VarN+"step");

  auto StartExpr = e.getStartExpr();

  //----------------------------------------------------------------------------
  // Dont create range
  if (auto RangeExpr = dynamic_cast<RangeExprAST*>(StartExpr)) {
    auto StartV = runExprVisitor(*RangeExpr->getStartExpr());
    StartV = TheHelper_.getAsValue(StartV);
    getBuilder().CreateStore(StartV, VarA);
    auto EndV = runExprVisitor(*RangeExpr->getEndExpr());
    EndV = TheHelper_.getAsValue(EndV);
    auto OneC = llvmValue<int_t>(getContext(), 1);
    EndV = getBuilder().CreateAdd(EndV, OneC);
    getBuilder().CreateStore(EndV, EndA);
    if (RangeExpr->hasStepExpr()) {
      auto StepV = runExprVisitor(*RangeExpr->getStepExpr());
      StepV = TheHelper_.getAsValue(StepV);
      getBuilder().CreateStore(StepV, StepA);
    }
    else {
      getBuilder().CreateStore(OneC, StepA);
    }
  }
  //----------------------------------------------------------------------------
  // Use pre-existing range
  else {
    auto RangeV = runStmtVisitor(*e.getStartExpr());
    auto StartV = Tasker_->getRangeStart(RangeV);
    getBuilder().CreateStore(StartV, VarA);
    auto EndV = Tasker_->getRangeEndPlusOne(RangeV);
    getBuilder().CreateStore(EndV, EndA);
    auto StepV = Tasker_->getRangeStep(RangeV);
    getBuilder().CreateStore(StepV, StepA);
  }

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *BeforeBB = BasicBlock::Create(getContext(), "beforeloop", TheFunction);
  BasicBlock *LoopBB =   BasicBlock::Create(getContext(), "loop", TheFunction);
  BasicBlock *IncrBB =   BasicBlock::Create(getContext(), "incr", TheFunction);
  BasicBlock *AfterBB =  BasicBlock::Create(getContext(), "afterloop", TheFunction);

  // set new exit block
  auto OldExitBlock = ExitBlock_;
  ExitBlock_ = AfterBB;

  getBuilder().CreateBr(BeforeBB);
  getBuilder().SetInsertPoint(BeforeBB);

  // Load value and check coondition
  Value *CurV = getBuilder().CreateLoad(VarT, VarA);

  // Compute the end condition.
  // Convert condition to a bool by comparing non-equal to 0.0.
  Value* EndV = getBuilder().CreateLoad(VarT, EndA);
  EndV = getBuilder().CreateICmpSLT(CurV, EndV, "loopcond");


  // Insert the conditional branch into the end of LoopEndBB.
  getBuilder().CreateCondBr(EndV, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(LoopBB);
  getBuilder().SetInsertPoint(LoopBB);
  
  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  createScope();
  bool HasBreak = 0;
  for ( auto & Stmt : e.getBodyExprs() ) {
    auto StmtPtr = Stmt.get();
    if (dynamic_cast<BreakStmtAST*>(StmtPtr)) HasBreak = true;
    runStmtVisitor(*StmtPtr);
    if (HasBreak) break;
  }
  popScope();


  // Insert unconditional branch to increment.
  if (!HasBreak) getBuilder().CreateBr(IncrBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  getBuilder().SetInsertPoint(IncrBB);
  

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  TheHelper_.increment( TheHelper_.getAsAlloca(VarA), StepA );

  // Insert the conditional branch into the end of LoopEndBB.
  getBuilder().CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  getBuilder().SetInsertPoint(AfterBB);

  // reset exit block
  ExitBlock_ = OldExitBlock;

  // for expr always returns 0.
  popScope();
  ValueResult_ = UndefValue::get(VoidType_);
}

//==============================================================================
// Foreach parallel loop
//==============================================================================
void CodeGen::visit(ForeachStmtAST& e)
{
  //---------------------------------------------------------------------------
  if (!e.isLifted()) {
    visit( static_cast<ForStmtAST&>(e) );
  }
  //---------------------------------------------------------------------------
  else {
    createScope();
    Tasker_->markTask(*TheModule_);
    
    auto TaskN = e.getName();
    const auto & TaskI = Tasker_->getTask(TaskN);
    
    //----------------------------------
    // Map partitions
    std::map<std::string, Value*> Ranges;
    std::map<std::string, Value*> Fields;
    for (auto & Stmt : e.getBodyExprs()) {
      auto StmtPtr = Stmt.get();
      if (auto Node = dynamic_cast<PartitionStmtAST*>(StmtPtr)) {
        auto VarA = runStmtVisitor(*Node->getPartExpr());
        auto NumVars = Node->getNumVars();
        for (unsigned i=0; i<NumVars; ++i) {
          auto VarD = Node->getVarDef(i);
          const auto VarN = Node->getVarName(i);
          if (VarD->getType().isRange()) {
            Ranges.emplace( VarN, VarA );
          }
          else if (VarD->getType().isField()) {
            Fields.emplace(VarN, VarA);
          }
        }
      }
    }

    //----------------------------------
    // Prepare arguments
    std::vector<Value*> TaskArgAs;
    std::vector<Value*> PartAs;
    for ( const auto & VarD : e.getAccessedVariables() ) {
      const auto & Name = VarD->getName();
      auto VarE = getVariable(Name);
      TaskArgAs.emplace_back(VarE->getAlloca());

      Value* PartA = nullptr;
      auto pit = Ranges.find(Name);
      auto fit = Fields.find(Name);
      if      (pit != Ranges.end()) PartA = pit->second;
      else if (fit != Fields.end()) PartA = fit->second;
      PartAs.emplace_back(PartA);
    }

    Value* RangeV = runStmtVisitor(*e.getStartExpr());

    //----------------------------------
    // Launch Task
	  
    if (e.hasReduction() && TaskI.hasReduction()) {
      auto FutureV = Tasker_->launch(
          *TheModule_,
          TaskI,
          TaskArgAs,
          PartAs,
          RangeV,
          TaskI.getReduction());
      
      std::vector<VariableType> ReduceTypes;
      std::vector<Value*> ReduceAs;
      for (auto ReduceD : e.getReductionVars()) {
        auto VarD = ReduceD.getVariableDef();
        ReduceTypes.emplace_back( VarD->getType() );
        auto VarE = getVariable( VarD->getName() );
        ReduceAs.emplace_back( VarE->getAlloca() );
      }

      auto ResultType = VariableType(ReduceTypes);
      auto ResultT = getLLVMType(ResultType);

      auto ResultV = Tasker_->loadFuture(*TheModule_, FutureV, ResultT);

      for (unsigned i=0; i<ReduceAs.size(); ++i) {
        //outs()<<"\n"; ResultV->print(outs()); outs()<<"\n";
        //outs()<<"\n"; ResultV->getType()->print(outs()); outs()<<"\n";
        auto ValueV = TheHelper_.extractValue(ResultV, i);
        getBuilder().CreateStore(ValueV, ReduceAs[i]);
      }
    }
    else {
      Tasker_->launch(*TheModule_, TaskI, TaskArgAs, PartAs, RangeV);
    }
    
    popScope();
    ValueResult_ = UndefValue::get(VoidType_);

    Tasker_->unmarkTask(*TheModule_);
  }
}

//==============================================================================
// Assignment - Many to one
//==============================================================================
void CodeGen::assignManyToOne(
    AssignStmtAST & e,
    const RightExprTuple & RightTuple)
{
  auto NumLeft = e.getNumLeftExprs();
    
  auto Right = std::get<0>(RightTuple);
  const auto & RightType = std::get<1>(RightTuple);
  auto RightExpr = std::get<2>(RightTuple);

  // Loop over variables
  for (unsigned i=0; i<NumLeft; i++) {

    auto RightV = Right;

    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    auto LeftExpr = dynamic_cast<VarAccessExprAST*>( e.getLeftExpr(i) );
    if (!LeftExpr)
      THROW_CONTRA_ERROR("Left-hand-side of expression is not a variable access.");
    
    // Codegen the RHS.
    if (auto CastType = e.getCast(i))
      RightV = TheHelper_.createCast(RightV, getLLVMType(*CastType));

    // Look up the name.
    const auto & VarType = LeftExpr->getType();
    const auto & VarN = LeftExpr->getName();
    auto VarPair = getOrCreateVariable(VarN, VarType);
    auto VarE = VarPair.first;
    auto VarInserted = VarPair.second; 
    auto VarA = VarE->getAlloca();
    
    // if the left side is not a future, then make sure the right side is loaded
    if (!Tasker_->isFuture(VarA) && Tasker_->isFuture(RightV)) {
      auto RightT = getLLVMType( setFuture(RightType, false) ); 
      Right = Tasker_->loadFuture(*TheModule_, RightV, RightT);
      RightV = Right;
    }

    //---------------------------------------------------------------------------
    // array[i] = scalar
    if (auto LeftAssignExpr = dynamic_cast<ArrayAccessExprAST*>(LeftExpr)) {
      auto IndexV = runExprVisitor(*LeftAssignExpr->getIndexExpr());
      if (Tasker_->isAccessor(VarA))
        Tasker_->storeAccessor(*TheModule_, RightV, VarA, IndexV);
      else if (Tasker_->isField(VarA)) {
        auto ElementT = getLLVMType(setField(VarType, false));
        Tasker_->createField(*TheModule_, VarA, VarN, ElementT, IndexV, RightV);
      }
      else
        insertArrayValue(VarA, VarE->getType(), IndexV, RightV);
    }
    //---------------------------------------------------------------------------
    // array = ?
    else if (isArray(VarA)) {
      // steal the alloca
      auto ArrayExpr = dynamic_cast<ArrayExprAST*>(RightExpr);
      if (i==0 && ArrayExpr) {
        if (!VarInserted) destroyVariable(*VarE);
        VarE->setAlloca(Right, true);
        eraseVariable(ArrayExpr->getName());
      }
      // otherwise, just copy it
      else {
        if (VarInserted) {
          auto SizeV = TheHelper_.extractValue(RightV, 1);
          allocateArray(VarA, SizeV, VarE->getType() );
        }
        copyArray(RightV, VarA);
        VarE->setOwner(true);
      }
    }
    //---------------------------------------------------------------------------
    // future = ?
    else if (Tasker_->isFuture(VarA)) {
      // future = future
      if (Tasker_->isFuture(RightV))
        Tasker_->copyFuture(*TheModule_, RightV, VarA);
      // future = value
      else
        Tasker_->toFuture(*TheModule_, RightV, VarA);
    }
    //---------------------------------------------------------------------------
    // Field = value
    else if (Tasker_->isAccessor(VarA)) {
      Tasker_->storeAccessor(*TheModule_, RightV, VarA);
    }
    //---------------------------------------------------------------------------
    // Range = ?
    else if (Tasker_->isRange(VarA)) {
      // steal the alloca
      auto RangeExpr = dynamic_cast<RangeExprAST*>(RightExpr);
      if (i==0 && RangeExpr) {
        if (!VarInserted) destroyVariable(*VarE);
        VarE->setAlloca(Right, true);
        eraseVariable(RangeExpr->getName());
      }
      // otherwise, just copy it
      else {
        RightV = TheHelper_.getAsValue(RightV);
        getBuilder().CreateStore(RightV, VarA);
        VarE->setOwner(false);
      }
    }
    //---------------------------------------------------------------------------
    // scalar = scalar
    else {
      RightV = TheHelper_.getAsValue(RightV);
      getBuilder().CreateStore(RightV, VarA);
    }

  } // for

}


//==============================================================================
// Assignment - Many to Many
//==============================================================================
void CodeGen::assignManyToMany(
    AssignStmtAST & e,
    const std::vector<RightExprTuple> & RightTuples)
{
  auto NumLeft = e.getNumLeftExprs();

  // Loop over variables
  for (unsigned i=0; i<NumLeft; i++) {
  
    auto RightV = std::get<0>(RightTuples[i]);
    const auto & RightType = std::get<1>(RightTuples[i]);
    auto RightExpr = std::get<2>(RightTuples[i]);


    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    auto LeftExpr = dynamic_cast<VarAccessExprAST*>( e.getLeftExpr(i) );
    if (!LeftExpr)
      THROW_CONTRA_ERROR("Left-hand-side of expression is not a variable access.");
    
    // Codegen the RHS.
    if (auto CastType = e.getCast(i))
      RightV = TheHelper_.createCast(RightV, getLLVMType(*CastType));

    // Look up the name.
    const auto & VarType = LeftExpr->getType();
    const auto & VarN =   LeftExpr->getName();
    auto VarPair = getOrCreateVariable(VarN, VarType);
    auto VarE = VarPair.first;
    auto VarInserted = VarPair.second; 
    auto VarA = VarE->getAlloca();
    
    // if the left side is not a future, then make sure the right side is loaded
    if (!Tasker_->isFuture(VarA) && Tasker_->isFuture(RightV)) {
      auto RightT = getLLVMType( setFuture(RightType, false) ); 
      RightV = Tasker_->loadFuture(*TheModule_, RightV, RightT);
    }

    //---------------------------------------------------------------------------
    // array[i] = scalar
    if (auto LeftAssignExpr = dynamic_cast<ArrayAccessExprAST*>(LeftExpr)) {
      auto IndexV = runExprVisitor(*LeftAssignExpr->getIndexExpr());
      if (Tasker_->isAccessor(VarA))
        Tasker_->storeAccessor(*TheModule_, RightV, VarA, IndexV);
      else if (Tasker_->isField(VarA)) {
        auto ElementT = getLLVMType(setField(VarType, false));
        Tasker_->createField(*TheModule_, VarA, VarN, ElementT, IndexV, RightV);
      }
      else
        insertArrayValue(VarA, VarE->getType(), IndexV, RightV);
    }
    //---------------------------------------------------------------------------
    // array = ?
    else if (isArray(VarA)) {
      // steal the alloca
      if (auto ArrayExpr = dynamic_cast<ArrayExprAST*>(RightExpr)) {
        if (!VarInserted) destroyVariable(*VarE);
        VarE->setAlloca(RightV, true);
        eraseVariable(ArrayExpr->getName());
      }
      else {
        if (VarInserted) {
          auto SizeV = TheHelper_.extractValue(RightV, 1);
          allocateArray(VarA, SizeV, VarE->getType() );
        }
        copyArray(RightV, VarA);
        VarE->setOwner(true);
      }
    }
    //---------------------------------------------------------------------------
    // future = ?
    else if (Tasker_->isFuture(VarA)) {
      // future = future
      if (Tasker_->isFuture(RightV))
        Tasker_->copyFuture(*TheModule_, RightV, VarA);
      // future = value
      else
        Tasker_->toFuture(*TheModule_, RightV, VarA);
    }
    //---------------------------------------------------------------------------
    // Field = value
    else if (Tasker_->isAccessor(VarA)) {
      Tasker_->storeAccessor(*TheModule_, RightV, VarA);
    }
    //---------------------------------------------------------------------------
    // Range = ?
    else if (Tasker_->isRange(VarA)) {
      // steal the alloca
      if (auto RangeExpr = dynamic_cast<RangeExprAST*>(RightExpr)) {
        if (!VarInserted) destroyVariable(*VarE);
        VarE->setAlloca(RightV, true);
        eraseVariable(RangeExpr->getName());
      }
      // otherwise, just copy it
      else {
        RightV = TheHelper_.getAsValue(RightV);
        getBuilder().CreateStore(RightV, VarA);
        VarE->setOwner(false);
      }
    }
    //---------------------------------------------------------------------------
    // scalar = scalar
    else {
      RightV = TheHelper_.getAsValue(RightV);
      getBuilder().CreateStore(RightV, VarA);
    }

  } // for

}

//==============================================================================
// Assignment
//==============================================================================
void CodeGen::visit(AssignStmtAST & e)
{
  auto NumLeft = e.getNumLeftExprs();
  auto NumRight = e.getNumRightExprs();
 
  // get all right 
  std::vector<RightExprTuple> RightTuples;
  for (const auto & Node : e.getRightExprs()) {
    auto Expr = static_cast<ExprAST*>(Node.get());
    if (!Expr)
      THROW_CONTRA_ERROR("Right-hand-side of expression is not am assignment.");
    RightTuples.emplace_back( runExprVisitor(*Expr), Expr->getType(), Expr );
  }

  if (IsPacked_) {
    auto StructV = std::get<0>(RightTuples.front());
    auto VarType = std::get<1>(RightTuples.front());
    auto Expr = std::get<2>(RightTuples.front());
    auto StructT = cast<StructType>(StructV->getType());
    NumRight = StructT->getNumElements();
    RightTuples.clear();
    for (unsigned i=0; i<NumRight; ++i) {
      RightTuples.emplace_back( 
          TheHelper_.extractValue(StructV, i),
          VarType.getMember(i),
          Expr
      );
    }
  }

  if ((NumLeft != NumRight) && (NumRight==1))
    assignManyToOne(e, RightTuples.front());
  else
    assignManyToMany(e, RightTuples);

  ValueResult_ = nullptr;
  return;

}

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function.
//==============================================================================
void CodeGen::visit(PrototypeAST &e) {

  unsigned NumArgs = e.getNumArgs();

  std::vector<Type *> ArgTypes;
  ArgTypes.reserve(NumArgs);

  for (unsigned i=0; i<NumArgs; ++i) {
    auto VarType = getLLVMType(e.getArgTypeId(i));
    if (e.isArgArray(i)) VarType = ArrayType_;
    ArgTypes.emplace_back(VarType);
  }
  
  Type* ReturnType = VoidType_;
  if (e.getReturnType()) {
    ReturnType = getLLVMType(e.getReturnType());
  }

  FunctionType *FT = FunctionType::get(ReturnType, ArgTypes, false);

  Function *F = Function::Create(FT, Function::ExternalLinkage, e.getName(), *TheModule_);

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args()) Arg.setName(e.getArgName(Idx++));

  FunctionResult_ = F;

}
 

//==============================================================================
/// FunctionAST Helper
//==============================================================================
Value* CodeGen::codegenFunctionBody(FunctionAST& e)
{
  for ( auto & stmt : e.getBodyExprs() )
  {
    runStmtVisitor(*stmt);
  }

  // Finish off the function.
  Value* RetVal = nullptr;
  if ( e.getReturnExpr() ) {
    RetVal = runStmtVisitor(*e.getReturnExpr());
    if (RetVal) RetVal = TheHelper_.getAsValue(RetVal);
  }

  return RetVal;
}

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
void CodeGen::visit(FunctionAST& e)
{

  bool CreatedScope = false;
  if (!e.isTopLevelExpression()) {
    CreatedScope = true;
    createScope();
  }

  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto & P = insertFunction( e.moveProtoExpr() );
  const auto & Name = P.getName();
  auto TheFunction = getFunction(Name).first;
  
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(getContext(), "entry", TheFunction);
  getBuilder().SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  unsigned ArgIdx = 0;
  for (auto &Arg : TheFunction->args()) {

    // get arg type
    auto ArgType = P.getArgType(ArgIdx);
    auto BaseType = strip(ArgType);
    auto LLType = getLLVMType(BaseType);

    // Create an alloca for this variable.
    VariableAlloca* VarE;
    if (ArgType.isArray()) {
      VarE = createArray(Arg.getName(), LLType);
    }
    else
      VarE = createVariable(Arg.getName(), LLType);
    VarE->setOwner(false);
    auto Alloca = VarE->getAlloca();
    
    // Store the initial value into the alloca.
    getBuilder().CreateStore(&Arg, Alloca);
 
    ArgIdx++;
  }
 
  // codegen the function body
  auto RetVal = codegenFunctionBody(e);

  // garbage collection
  if (CreatedScope) popScope();
  
  // Finish off the function.
  if (RetVal && !RetVal->getType()->isVoidTy() ) {
    getBuilder().CreateRet(RetVal);
  }
  else
    getBuilder().CreateRetVoid();
  
  // Validate the generated code, checking for consistency.
  verifyFunction(*TheFunction);
    
  if (DeviceJIT_)
    DeviceJIT_->addModule( TheModule_.get() );
  
  FunctionResult_ = TheFunction;

}

//==============================================================================
/// TaskAST - This class represents a function definition itself.
//==============================================================================
void CodeGen::visit(TaskAST& e)
{
  bool CreatedScope = false;
  if (!e.isTopLevelExpression()) {
    CreatedScope = true;
    createScope();
  }
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto & P = insertFunction( e.moveProtoExpr() );
  auto Name = P.getName();
  auto TheFunction = getFunction(Name).first;
  
  // generate wrapped task
  auto Wrapper = Tasker_->taskPreamble(*TheModule_, Name, TheFunction);
  
  // insert the task 
  auto & TaskI = Tasker_->insertTask(Name, Wrapper.TheFunction);
  TaskI.setLeaf(e.isLeaf());

  // insert arguments into variable table
  unsigned ArgIdx = 0;
  for (auto &Arg : TheFunction->args()) {
    auto Alloca = Wrapper.ArgAllocas[ArgIdx++];
    auto AllocaT = Alloca->getAllocatedType(); // FIX FOR ARRAYS
    auto VarE = insertVariable(Arg.getName(), Alloca, AllocaT);
    VarE->setOwner(false);
  }

  
  // codegen the function body
  auto RetVal = codegenFunctionBody(e);
  
  if (RetVal && Tasker_->isFuture(RetVal)) {
    auto RetT = getLLVMType( P.getReturnType() );
    RetVal = Tasker_->loadFuture(*TheModule_, RetVal, RetT);
  }

  // garbage collection
  if (CreatedScope) popScope();

  // Finish wrapped task
  Tasker_->taskPostamble(*TheModule_, RetVal, false);
  
  // Validate the generated code, checking for consistency.
  verifyFunction(*Wrapper.TheFunction);
  
  // set the finished function
  FunctionResult_ = Wrapper.TheFunction;

}

//==============================================================================
/// TaskAST - This class represents a function definition itself.
//==============================================================================
void CodeGen::visit(IndexTaskAST& e)
{
  auto TaskN = e.getName();

  // create a new device module
  std::unique_ptr<Module> OldModule;
  if (DeviceJIT_) {
    OldModule = std::move(TheModule_);
    TheModule_ = DeviceJIT_->createModule();
  }

  //----------------------------------------------------------------------------
  // Reduction Op
  Type* ResultT = nullptr;
  std::unique_ptr<AbstractReduceInfo> RedopInfo;

  if (e.hasReduction()) {
   
    std::vector<VariableType> ReduceTypes;
    std::vector<Type*> ReduceTs;
    std::vector<ReductionType> ReduceOps;

    for (auto ReduceD : e.getReductionDefs()) {
      auto VarD = ReduceD.getVariableDef();
      auto VarType = VarD->getType();
      ReduceTs.emplace_back(getLLVMType(VarType));
      ReduceTypes.emplace_back(VarType);
      ReduceOps.emplace_back(ReduceD.getType());
    }
    auto ReturnType = VariableType(ReduceTypes);
    ResultT = getLLVMType(ReturnType);

    auto ModulePtr = (DeviceJIT_) ?
      new Module("temporary module", getContext()) : TheModule_.get(); 

    RedopInfo = Tasker_->createReductionOp(
          *ModulePtr,
          TaskN,
          ReduceTs,
          ReduceOps);

    if (DeviceJIT_) {
      utils::insertModule(*ModulePtr, *TheModule_);
      utils::insertModule(*ModulePtr, *OldModule);
      delete ModulePtr;
    }


  }

  //----------------------------------------------------------------------------
  // Index Task

  bool CreatedScope = false;
  if (!e.isTopLevelExpression()) {
    CreatedScope = true;
    createScope();
  }

  // get global args
  std::vector<Type*> TaskArgTs;
  std::vector<std::string> TaskArgNs;
  for ( const auto & VarE : e.getVariableDefs() ) {
    const auto & VarN = VarE->getName();
    TaskArgNs.emplace_back( VarN );
    // overrided field types
    const auto & VarT = VarE->getType();
    if (VarT.isField())
      TaskArgTs.emplace_back( AccessorType_ ); 
    else
      TaskArgTs.emplace_back( getLLVMType(VarT) ); 
  }
      
	// generate wrapped task
  auto Wrapper = Tasker_->taskPreamble(
      *TheModule_,
      TaskN, 
      TaskArgNs,
      TaskArgTs,
      ResultT);

  // insert arguments into variable table
  for (unsigned ArgIdx=0; ArgIdx<TaskArgNs.size(); ++ArgIdx) {
    auto VarA = Wrapper.ArgAllocas[ArgIdx];
    auto AllocaT = VarA->getAllocatedType(); // FIX FOR ARRAYS
    auto VarD = e.getVariableDef(ArgIdx);
    bool IsOwner = false;
    if (Tasker_->isAccessor(AllocaT)) IsOwner = true;
    auto VarT = getLLVMType( strip(VarD->getType()) );
    auto VarE = insertVariable(TaskArgNs[ArgIdx], VarA, VarT);
    VarE->setOwner(IsOwner);
  }
  
  // and the index
  auto IndexA = Wrapper.Index;
  auto IndexT = IndexA->getAllocatedType();
  auto IndexE = insertVariable(e.getLoopVariableName(), IndexA, IndexT);
  IndexE->setOwner(false);

  //// function body
  for ( auto & stmt : e.getBodyExprs() ) runStmtVisitor(*stmt);
  
  // if reductions
  Value* ResultA = nullptr;
  if (ResultT) {
    ResultA = TheHelper_.createEntryBlockAlloca(ResultT, "reduce.a");
    unsigned i=0;
    for (auto VarD : e.getReductionDefs()) {
      const auto & VarN = VarD.getVariableDef()->getName();
      auto VarE = getVariable(VarN);
      auto MemberA = VarE->getAlloca();
      TheHelper_.insertValue(ResultA, MemberA, i);
      i++;
    } 
  }
  
  // garbage collection
  if (CreatedScope) popScope();
  
  // finish task
  Tasker_->taskPostamble(*TheModule_, ResultA, true);
  
	// register it
  auto & TaskI = Tasker_->insertTask(TaskN, Wrapper.TheFunction);
  TaskI.setLeaf(e.isLeaf());
  if (RedopInfo) TaskI.setReduction( std::move(RedopInfo) );

 	verifyFunction(*Wrapper.TheFunction);

  // Compile device code
  if (DeviceJIT_) {
    DeviceJIT_->addModule( std::move(TheModule_) );
    TheModule_ = std::move(OldModule);
  }

	FunctionResult_ = Wrapper.TheFunction;
}

} // namespace
