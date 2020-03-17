#include "ast.hpp"
#include "codegen.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "precedence.hpp"
#include "token.hpp"

#include "librt/librt.hpp"

#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

using namespace llvm;

namespace contra {

//==============================================================================
// Constructor
//==============================================================================
CodeGen::CodeGen (bool debug = false) : Builder_(TheContext_)
{

  I64Type_  = llvmIntegerType(TheContext_);
  F64Type_  = llvmRealType(TheContext_);
  VoidType_ = Type::getVoidTy(TheContext_);

  TypeTable_.emplace( Context::I64Type->getName(),  I64Type_);
  TypeTable_.emplace( Context::F64Type->getName(),  F64Type_);
  TypeTable_.emplace( Context::VoidType->getName(), VoidType_);

  initializeModuleAndPassManager();

  if (debug) {

    // Add the current debug info version into the module.
    TheModule_->addModuleFlag(Module::Warning, "Debug Info Version",
                             DEBUG_METADATA_VERSION);

    // Darwin only supports dwarf2.
    if (Triple(sys::getProcessTriple()).isOSDarwin())
      TheModule_->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);

    DBuilder = std::make_unique<DIBuilder>(*TheModule_);
    KSDbgInfo.TheCU = DBuilder->createCompileUnit(
      dwarf::DW_LANG_C, DBuilder->createFile("fib.ks", "."),
      "Kaleidoscope Compiler", 0, "", 0);
  }

}

//==============================================================================
// Initialize module and optimizer
//==============================================================================
void CodeGen::initializeModuleAndPassManager() {
  initializeModule();
  if (!isDebug())
    initializePassManager();
}


//==============================================================================
// Initialize module
//==============================================================================
void CodeGen::initializeModule() {

  // Open a new module.
  TheModule_ = std::make_unique<Module>("my cool jit", TheContext_);
  TheModule_->setDataLayout(TheJIT_.getTargetMachine().createDataLayout());

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

//==============================================================================
// Get the function
//==============================================================================
Function *CodeGen::getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto F = TheModule_->getFunction(Name))
    return F;
  
  // see if this is an available intrinsic, try installing it first
  if (auto F = librt::RunTimeLib::tryInstall(TheContext_, *TheModule_, Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto fit = FunctionTable_.find(Name);
  return runFuncVisitor(*fit->second);
}

PrototypeAST & CodeGen::insertFunction(std::unique_ptr<PrototypeAST> Proto)
{
  auto & P = FunctionTable_[Proto->getName()];
  P = std::move(Proto);
  return *P;
}

//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
AllocaInst *CodeGen::createEntryBlockAlloca(Function *TheFunction,
    const std::string &VarName, Type* VarType)
{
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(VarType, nullptr, VarName.c_str());
}

AllocaInst *CodeGen::createVariable(Function *TheFunction,
  const std::string &VarName, Type* VarType)
{
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
  auto Alloca = TmpB.CreateAlloca(VarType, nullptr, VarName.c_str());
  VariableTable_[VarName] = Alloca;
  return Alloca;
}
    
//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
ArrayType
CodeGen::createArray(Function *TheFunction, const std::string &VarName,
    Type * ElementType, Value * SizeExpr)
{

  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());

  Function *F; 
  F = TheModule_->getFunction("allocate");
  if (!F) F = librt::RunTimeLib::tryInstall(TheContext_, *TheModule_, "allocate");


  auto PtrType = PointerType::get(ElementType, 0);
  auto TheBlock = Builder_.GetInsertBlock();
  auto Index = ConstantInt::get(TheContext_, APInt(32, 1, true));
  auto Null = Constant::getNullValue(PtrType);
  auto SizeGEP = Builder_.CreateGEP(ElementType, Null, Index, "size");
  auto DataSize = CastInst::Create(Instruction::PtrToInt, SizeGEP,
          llvmIntegerType(TheContext_), "sizei", TheBlock);

  auto TotalSize = Builder_.CreateMul(SizeExpr, DataSize, "multmp");

  Value* CallInst = Builder_.CreateCall(F, TotalSize, VarName+"vectmp");
  auto ResType = CallInst->getType();
  auto AllocInst = TmpB.CreateAlloca(ResType, 0, VarName+"vec");
  Builder_.CreateStore(CallInst, AllocInst);

  std::vector<Value*> MemberIndices(2);
  MemberIndices[0] = ConstantInt::get(TheContext_, APInt(32, 0, true));
  MemberIndices[1] = ConstantInt::get(TheContext_, APInt(32, 0, true));

  auto GEPInst = Builder_.CreateGEP(ResType, AllocInst, MemberIndices,
      VarName+"vec.ptr");
  auto LoadedInst = Builder_.CreateLoad(GEPInst->getType()->getPointerElementType(),
      GEPInst, VarName+"vec.val");

  Value* Cast = CastInst::Create(CastInst::BitCast, LoadedInst, PtrType, "casttmp", TheBlock);

  ArrayTable_[VarName] = ArrayType{AllocInst, Cast, SizeExpr};
  return ArrayTable_[VarName];
}
 
//==============================================================================
// Initialize Array
//==============================================================================
void CodeGen::initArrays( Function *TheFunction,
    const std::vector<AllocaInst*> & VarList,
    Value * InitVal,
    Value * SizeExpr )
{

  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
  
  auto Alloca = TmpB.CreateAlloca(llvmIntegerType(TheContext_), nullptr, "__i");
  Value * StartVal = llvmInteger(TheContext_, 0);
  Builder_.CreateStore(StartVal, Alloca);
  
  auto BeforeBB = BasicBlock::Create(TheContext_, "beforeinit", TheFunction);
  auto LoopBB =   BasicBlock::Create(TheContext_, "init", TheFunction);
  auto AfterBB =  BasicBlock::Create(TheContext_, "afterinit", TheFunction);
  Builder_.CreateBr(BeforeBB);
  Builder_.SetInsertPoint(BeforeBB);
  auto CurVar = Builder_.CreateLoad(llvmIntegerType(TheContext_), Alloca);
  auto EndCond = Builder_.CreateICmpSLT(CurVar, SizeExpr, "initcond");
  Builder_.CreateCondBr(EndCond, LoopBB, AfterBB);
  Builder_.SetInsertPoint(LoopBB);

  for ( auto i : VarList) {
    auto LoadType = i->getType()->getPointerElementType();
    auto Load = Builder_.CreateLoad(LoadType, i, "ptr"); 
    auto GEP = Builder_.CreateGEP(Load, CurVar, "offset");
    Builder_.CreateStore(InitVal, GEP);
  }

  auto StepVal = llvmInteger(TheContext_, 1);
  auto NextVar = Builder_.CreateAdd(CurVar, StepVal, "nextvar");
  Builder_.CreateStore(NextVar, Alloca);
  Builder_.CreateBr(BeforeBB);
  Builder_.SetInsertPoint(AfterBB);
}
  
//==============================================================================
// Initialize an arrays individual values
//==============================================================================
void CodeGen::initArray(
    Function *TheFunction, 
    AllocaInst* Var,
    const std::vector<Value *> InitVals )
{
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
  
  auto NumVals = InitVals.size();
  auto TheBlock = Builder_.GetInsertBlock();

  auto LoadType = Var->getType()->getPointerElementType();
  auto ValType = LoadType->getPointerElementType();
  auto Load = Builder_.CreateLoad(LoadType, Var, "ptr"); 
  
  for (std::size_t i=0; i<NumVals; ++i) {
    auto Index = llvmInteger(TheContext_, i);
    auto GEP = Builder_.CreateGEP(Load, Index, "offset");
    auto Init = InitVals[i];
    auto InitType = Init->getType();
    if ( InitType->isFloatingPointTy() && ValType->isIntegerTy() ) {
      auto Cast = CastInst::Create(Instruction::FPToSI, Init,
          llvmIntegerType(TheContext_), "cast", TheBlock);
      Init = Cast;
    }
    else if ( InitType->isIntegerTy() && ValType->isFloatingPointTy() ) {
      auto Cast = CastInst::Create(Instruction::SIToFP, Init,
          llvmRealType(TheContext_), "cast", TheBlock);
      Init = Cast;
    }
    else if (InitType!=ValType)
      THROW_CONTRA_ERROR("Unknown cast operation");
    Builder_.CreateStore(Init, GEP);
  }
}

//==============================================================================
// Copy Array
//==============================================================================
void CodeGen::copyArrays(
    Function *TheFunction,
    AllocaInst* Src,
    const std::vector<AllocaInst*> Tgts,
    Value * NumElements)
{

  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());

  auto Alloca = TmpB.CreateAlloca(llvmIntegerType(TheContext_), nullptr, "__i");
  Value * StartVal = llvmInteger(TheContext_, 0);
  Builder_.CreateStore(StartVal, Alloca);
  
  auto BeforeBB = BasicBlock::Create(TheContext_, "beforeinit", TheFunction);
  auto LoopBB =   BasicBlock::Create(TheContext_, "init", TheFunction);
  auto AfterBB =  BasicBlock::Create(TheContext_, "afterinit", TheFunction);
  Builder_.CreateBr(BeforeBB);
  Builder_.SetInsertPoint(BeforeBB);
  auto CurVar = Builder_.CreateLoad(llvmIntegerType(TheContext_), Alloca);
  auto EndCond = Builder_.CreateICmpSLT(CurVar, NumElements, "initcond");
  Builder_.CreateCondBr(EndCond, LoopBB, AfterBB);
  Builder_.SetInsertPoint(LoopBB);
    
  auto PtrType = Src->getType()->getPointerElementType();
  auto ValType = PtrType->getPointerElementType();

  auto SrcLoad = Builder_.CreateLoad(PtrType, Src, "srcptr"); 
  auto SrcGEP = Builder_.CreateGEP(SrcLoad, CurVar, "srcoffset");
  auto SrcVal = Builder_.CreateLoad(ValType, SrcLoad, "srcval");

  for ( auto T : Tgts ) {
    auto TgtLoad = Builder_.CreateLoad(PtrType, T, "tgtptr"); 
    auto TgtGEP = Builder_.CreateGEP(TgtLoad, CurVar, "tgtoffset");
    Builder_.CreateStore(SrcVal, TgtGEP);
  }

  auto StepVal = llvmInteger(TheContext_, 1);
  auto NextVar = Builder_.CreateAdd(CurVar, StepVal, "nextvar");
  Builder_.CreateStore(NextVar, Alloca);
  Builder_.CreateBr(BeforeBB);
  Builder_.SetInsertPoint(AfterBB);
}
  

//==============================================================================
// Destroy all arrays
//==============================================================================
void CodeGen::destroyArrays() {
  
  Function *F; 
  F = TheModule_->getFunction("deallocate");
  if (!F) F = librt::RunTimeLib::tryInstall(TheContext_, *TheModule_, "deallocate");
  
  for ( auto & i : NamedArrays )
  {
    const auto & Name = i.first;
    auto & Alloca = i.second;
    auto AllocaT = Alloca->getType()->getPointerElementType();
    auto Vec = Builder_.CreateLoad(AllocaT, Alloca, Name+"vec");
    Builder_.CreateCall(F, Vec, Name+"dealloctmp");
  }

  NamedArrays.clear();
}


//==============================================================================
// JIT the current module
//==============================================================================
JIT::VModuleKey CodeGen::doJIT()
{
  auto H = TheJIT_.addModule(std::move(TheModule_));
  initializeModuleAndPassManager();
  return H;
}

//==============================================================================
// Search the JIT for a symbol
//==============================================================================
JIT::JITSymbol CodeGen::findSymbol( const char * Symbol )
{ return TheJIT_.findSymbol(Symbol); }

//==============================================================================
// Delete a JITed module
//==============================================================================
void CodeGen::removeJIT( JIT::VModuleKey H )
{ TheJIT_.removeModule(H); }

//==============================================================================
// Finalize whatever needs to be
//==============================================================================
void CodeGen::finalize() {
  if (DBuilder) DBuilder->finalize();
}

//==============================================================================
// Create a debug function type
//==============================================================================
DISubroutineType *CodeGen::createFunctionType(unsigned NumArgs, DIFile *Unit) {
  
  if (!isDebug()) return nullptr;

  SmallVector<Metadata *, 8> EltTys;
  DIType *DblTy = KSDbgInfo.getDoubleTy(*DBuilder);

  // Add the result type.
  EltTys.push_back(DblTy);

  for (unsigned i = 0, e = NumArgs; i != e; ++i)
    EltTys.push_back(DblTy);

  return DBuilder->createSubroutineType(DBuilder->getOrCreateTypeArray(EltTys));
}
  
DIFile * CodeGen::createFile() {
  if (isDebug())
    return DBuilder->createFile(KSDbgInfo.TheCU->getFilename(),
        KSDbgInfo.TheCU->getDirectory());
  else
    return nullptr;
}

//==============================================================================
// create a subprogram DIE
//==============================================================================
DISubprogram * CodeGen::createSubprogram(
    unsigned LineNo,
    unsigned ScopeLine,
    const std::string & Name,
    unsigned arg_size,
    DIFile * Unit)
{
  if (isDebug()) {
    DIScope *FContext = Unit;
    DISubprogram *SP = DBuilder->createFunction(
        FContext, Name, StringRef(), Unit, LineNo,
        createFunctionType(arg_size, Unit), ScopeLine,
        DINode::FlagPrototyped, DISubprogram::SPFlagDefinition);
    return SP;
  }
  else {
    return nullptr;
  }
}
 
//==============================================================================
// Create a variable
//==============================================================================
DILocalVariable *CodeGen::createVariable( DISubprogram *SP,
    const std::string & Name, unsigned ArgIdx, DIFile *Unit, unsigned LineNo,
    AllocaInst *Alloca)
{
  if (isDebug()) {
    DILocalVariable *D = DBuilder->createParameterVariable(
      SP, Name, ArgIdx, Unit, LineNo, KSDbgInfo.getDoubleTy(*DBuilder),
      true);

    DBuilder->insertDeclare(Alloca, D, DBuilder->createExpression(),
        DebugLoc::get(LineNo, 0, SP),
        Builder_.GetInsertBlock());

    return D;
  }
  else {
    return nullptr;
  }
}


//==============================================================================
// IntegerExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
void CodeGen::dispatch(ValueExprAST<int_t> & e)
{
  emitLocation(&e);
  ValueResult_ = llvmInteger(TheContext_, e.Val_);
}

//==============================================================================
// RealExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
void CodeGen::dispatch(ValueExprAST<real_t> & e)
{
  emitLocation(&e);
  ValueResult_ = llvmReal(TheContext_, e.Val_);
}

//==============================================================================
// StringExprAST - Expression class for string literals like "hello".
//==============================================================================
void CodeGen::dispatch(ValueExprAST<std::string>& e)
{
  emitLocation(&e);
  auto ConstantArray = ConstantDataArray::getString(TheContext_, e.Val_);
  auto GVStr = new GlobalVariable(getModule(), ConstantArray->getType(), true,
      GlobalValue::InternalLinkage, ConstantArray);
  Constant* zero = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  Constant* strVal = ConstantExpr::getGetElementPtr(
      IntegerType::getInt8Ty(TheContext_), GVStr, zero, true);
  ValueResult_ = strVal;
}
 
//==============================================================================
// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
void CodeGen::dispatch(VariableExprAST& e)
{

  auto Name = e.getName();

  // Look this variable up in the function.
  Value* V = getVariable(e.getName());
  
  // Load the value.
  auto Ty = V->getType();
  Ty = Ty->getPointerElementType();
    
  auto Load = Builder_.CreateLoad(Ty, V, "val."+Name);

  if ( e.isArray() ) {
    Ty = Ty->getPointerElementType();
    auto IndexVal = runExprVisitor(*e.IndexExpr_);
    auto GEP = Builder_.CreateGEP(Load, IndexVal, Name+".offset");
    ValueResult_ = Builder_.CreateLoad(Ty, GEP, Name+".i");
  }
  else {
    ValueResult_ = Load;
  }
}

//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
void CodeGen::dispatch(ArrayExprAST &e)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // the llvm variable type
  auto VarType = getLLVMType(e.getType());
  auto VarPointerType = PointerType::get(VarType, 0);

  std::vector<Value*> InitVals;
  InitVals.reserve(e.ValExprs_.size());
  for ( auto & E : e.ValExprs_ ) 
    InitVals.emplace_back( runExprVisitor(*E) );

  Value* SizeExpr = nullptr;
  if (e.SizeExpr_) {
    SizeExpr = runExprVisitor(*e.SizeExpr_);
  }
  else {
    SizeExpr = llvmInteger(TheContext_, e.ValExprs_.size());
  }

  auto Array = createArray(TheFunction, "__tmp", VarType, SizeExpr );
  auto Alloca = createVariable(TheFunction, "__tmp", VarPointerType);
  Builder_.CreateStore(Array.Data, Alloca);

  if (e.SizeExpr_) 
    initArrays(TheFunction, {Alloca}, InitVals[0], SizeExpr);
  else
    initArray(TheFunction, Alloca, InitVals);

  ValueResult_ =  Alloca;
}
  
//==============================================================================
// CastExprAST - Expression class for casts.
//==============================================================================
void CodeGen::dispatch(CastExprAST &e)
{
  auto FromVal = runExprVisitor(*e.FromExpr_);
  auto FromType = ValueResult_->getType();

  auto ToType = getLLVMType(e.getType());

  auto TheBlock = Builder_.GetInsertBlock();

  if (FromType->isFloatingPointTy() && ToType->isIntegerTy()) {
    ValueResult_ = CastInst::Create(Instruction::FPToSI, FromVal,
        llvmRealType(TheContext_), "cast", TheBlock);
  }
  else if (FromType->isIntegerTy() && ToType->isFloatingPointTy()) {
    ValueResult_ = CastInst::Create(Instruction::SIToFP, FromVal,
        llvmIntegerType(TheContext_), "cast", TheBlock);
  }
  else {
    ValueResult_ = FromVal;
  }

}

//==============================================================================
// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
void CodeGen::dispatch(UnaryExprAST & e) {
  auto OperandV = runExprVisitor(*e.OpExpr_); 
  
  if (OperandV->getType()->isFloatingPointTy()) {
  
    switch (e.OpCode_) {
    case tok_sub:
      ValueResult_ = Builder_.CreateFNeg(OperandV, "negtmp");
      return;
    }

  }
  else {
    switch (e.OpCode_) {
    case tok_sub:
      ValueResult_ = Builder_.CreateNeg(OperandV, "negtmp");
      return;
    }
  }

  auto F = getFunction(std::string("unary") + e.OpCode_);
  emitLocation(&e);
  ValueResult_ = Builder_.CreateCall(F, OperandV, "unop");
}

//==============================================================================
// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
void CodeGen::dispatch(BinaryExprAST& e) {
  emitLocation(&e);
  
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (e.OpCode_ == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    auto LHSE = dynamic_cast<VariableExprAST*>(e.LeftExpr_.get());
    // Codegen the RHS.
    auto Val = runExprVisitor(*e.RightExpr_);

    // Look up the name.
    const auto & VarName = LHSE->getName();
    Value *Variable = getVariable(VarName);

    // array element access
    if (Variable->getType()->getPointerElementType()->isPointerTy()) {
      auto Ty = Variable->getType()->getPointerElementType();
      auto Load = Builder_.CreateLoad(Ty, Variable, "ptr."+VarName);
      auto IndexVal = runExprVisitor(*LHSE->IndexExpr_);
      auto GEP = Builder_.CreateGEP(Load, IndexVal, VarName+"aoffset");
      Builder_.CreateStore(Val, GEP);
    }
    else {
      Builder_.CreateStore(Val, Variable);
    }
    ValueResult_ = Val;
    return;
  }

  Value *L = runExprVisitor(*e.LeftExpr_);
  Value *R = runExprVisitor(*e.RightExpr_);

  auto l_is_real = L->getType()->isFloatingPointTy();
  auto r_is_real = R->getType()->isFloatingPointTy();
  bool is_real =  (l_is_real && r_is_real);

  if (is_real) {
    switch (e.OpCode_) {
    case tok_add:
      ValueResult_ = Builder_.CreateFAdd(L, R, "addtmp");
      return;
    case tok_sub:
      ValueResult_ = Builder_.CreateFSub(L, R, "subtmp");
      return;
    case tok_mul:
      ValueResult_ = Builder_.CreateFMul(L, R, "multmp");
      return;
    case tok_div:
      ValueResult_ = Builder_.CreateFDiv(L, R, "divtmp");
      return;
    case tok_lt:
      ValueResult_ = Builder_.CreateFCmpULT(L, R, "cmptmp");
      return;
    } 
  }
  else {
    switch (e.OpCode_) {
    case tok_add:
      ValueResult_ = Builder_.CreateAdd(L, R, "addtmp");
      return;
    case tok_sub:
      ValueResult_ = Builder_.CreateSub(L, R, "subtmp");
      return;
    case tok_mul:
      ValueResult_ = Builder_.CreateMul(L, R, "multmp");
      return;
    case tok_div:
      ValueResult_ = Builder_.CreateSDiv(L, R, "divtmp");
      return;
    case tok_lt:
      ValueResult_ = Builder_.CreateICmpSLT(L, R, "cmptmp");
      return;
    }
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  auto F = getFunction(std::string("binary") + e.OpCode_);

  Value *Ops[] = { L, R };
  ValueResult_ = Builder_.CreateCall(F, Ops, "binop");
}

//==============================================================================
// CallExprAST - Expression class for function calls.
//==============================================================================
void CodeGen::dispatch(CallExprAST &e) {
  emitLocation(&e);

  // Look up the name in the global module table.
  auto CalleeF = getFunction(e.Callee_);
  //auto FunType = CalleeF->getFunctionType();
  //auto NumFixedArgs = FunType->getNumParams();

  std::vector<Value *> ArgsV;
  for (unsigned i = 0; i<e.ArgExprs_.size(); ++i) {
    auto A = runExprVisitor(*e.ArgExprs_[i]);
    ArgsV.push_back(A);
  }

  ValueResult_ = Builder_.CreateCall(CalleeF, ArgsV, "calltmp");
}

//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
void CodeGen::dispatch(IfStmtAST & e) {
  emitLocation(&e);

  if ( e.ThenExpr_.empty() && e.ElseExpr_.empty() ) {
    ValueResult_ = Constant::getNullValue(VoidType_);
    return;
  }

  Value *CondV = runExprVisitor(*e.CondExpr_);

  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(TheContext_, "then", TheFunction);
  BasicBlock *ElseBB = e.ElseExpr_.empty() ? nullptr : BasicBlock::Create(TheContext_, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheContext_, "ifcont");

  if (ElseBB)
    Builder_.CreateCondBr(CondV, ThenBB, ElseBB);
  else
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);

  // Emit then value.
  Builder_.SetInsertPoint(ThenBB);

  for ( auto & stmt : e.ThenExpr_ ) runExprVisitor(*stmt);

  // get first non phi instruction
  auto ThenV = ThenBB->getFirstNonPHI();

  Builder_.CreateBr(MergeBB);

  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder_.GetInsertBlock();

  if (ElseBB) {

    // Emit else block.
    TheFunction->getBasicBlockList().push_back(ElseBB);
    Builder_.SetInsertPoint(ElseBB);

    for ( auto & stmt : e.ElseExpr_ ) runExprVisitor(*stmt);

    // get first non phi
    auto ElseV = ElseBB->getFirstNonPHI();

    Builder_.CreateBr(MergeBB);
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = Builder_.GetInsertBlock();

  } // else

  // Emit merge block.
  TheFunction->getBasicBlockList().push_back(MergeBB);
  Builder_.SetInsertPoint(MergeBB);
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
void CodeGen::dispatch(ForStmtAST& e) {
  
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  auto LLType = llvmIntegerType(TheContext_);
  AllocaInst *Alloca = createVariable(TheFunction, e.VarId_.getName(), LLType);
  
  emitLocation(&e);

  // Emit the start code first, without 'variable' in scope.
  auto StartVal = runExprVisitor(*e.StartExpr_);
  if (StartVal->getType()->isFloatingPointTy())
    THROW_IMPLEMENTED_ERROR("Cast required for start value");

  // Store the value into the alloca.
  Builder_.CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *BeforeBB = BasicBlock::Create(TheContext_, "beforeloop", TheFunction);
  BasicBlock *LoopBB =   BasicBlock::Create(TheContext_, "loop", TheFunction);
  BasicBlock *IncrBB =   BasicBlock::Create(TheContext_, "incr", TheFunction);
  BasicBlock *AfterBB =  BasicBlock::Create(TheContext_, "afterloop", TheFunction);

  Builder_.CreateBr(BeforeBB);
  Builder_.SetInsertPoint(BeforeBB);

  // Load value and check coondition
  Value *CurVar = Builder_.CreateLoad(LLType, Alloca);

  // Compute the end condition.
  // Convert condition to a bool by comparing non-equal to 0.0.
  auto EndCond = runExprVisitor(*e.EndExpr_);
  if (EndCond->getType()->isFloatingPointTy())
    THROW_IMPLEMENTED_ERROR("Cast required for end condition");
 
  if (e.Loop_ == ForStmtAST::LoopType::Until) {
    Value *One = llvmInteger(TheContext_, 1);
    EndCond = Builder_.CreateSub(EndCond, One, "loopsub");
  }

  EndCond = Builder_.CreateICmpSLE(CurVar, EndCond, "loopcond");


  // Insert the conditional branch into the end of LoopEndBB.
  Builder_.CreateCondBr(EndCond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(LoopBB);
  Builder_.SetInsertPoint(LoopBB);
  
  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  for ( auto & stmt : e.BodyExprs_ ) runExprVisitor(*stmt);

  // Insert unconditional branch to increment.
  Builder_.CreateBr(IncrBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  Builder_.SetInsertPoint(IncrBB);
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (e.StepExpr_) {
    StepVal = runExprVisitor(*e.StepExpr_);
    if (StepVal->getType()->isFloatingPointTy())
      THROW_IMPLEMENTED_ERROR("Cast required for step value");
  } else {
    // If not specified, use 1.0.
    StepVal = llvmInteger(TheContext_, 1);
  }


  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  CurVar = Builder_.CreateLoad(LLType, Alloca);
  Value *NextVar = Builder_.CreateAdd(CurVar, StepVal, "nextvar");
  Builder_.CreateStore(NextVar, Alloca);

  // Insert the conditional branch into the end of LoopEndBB.
  Builder_.CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  Builder_.SetInsertPoint(AfterBB);

  // for expr always returns 0.
  ValueResult_ = UndefValue::get(VoidType_);
}

//==============================================================================
// VarDefExprAST - Expression class for var/in
//==============================================================================
void CodeGen::dispatch(VarDeclAST & e) {
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.

  // Emit initializer first
  auto InitVal = runExprVisitor(*e.InitExpr_);
  //auto IType = InitVal->getType();
  
  // the llvm variable type
  auto VarType = getLLVMType(e.getType());

  // Register all variables and emit their initializer.
  for (const auto & VarId : e.VarIds_) {  
    auto Alloca = createVariable(TheFunction, VarId.getName(), VarType);
    Builder_.CreateStore(InitVal, Alloca);
  }

  emitLocation(&e);
  ValueResult_ = InitVal;
}

//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
void CodeGen::dispatch(ArrayDeclAST &e) {
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.

  
  // the llvm variable type
  auto VarType = getLLVMType(e.getType());
  auto VarPointerType = PointerType::get(VarType, 0);
    
  int NumVars = e.VarIds_.size();

  //---------------------------------------------------------------------------
  // Array already on right hand side
  auto ArrayAST = dynamic_cast<ArrayExprAST*>(e.InitExpr_.get());
  if (ArrayAST) {

    runExprVisitor(*ArrayAST);

    // transfer to first
    const auto & VarName = e.VarIds_[0].getName();
    auto Array = moveArray("__tmp", VarName);
    auto Alloca = moveVariable("__tmp", VarName);
    
    auto SizeExpr = Array.Size;
    ValueResult_ = Array.Data;
  
    // Register all variables and emit their initializer.
    std::vector<AllocaInst*> ArrayAllocas;
    for (int i=1; i<NumVars; ++i) {
      const auto & VarName = e.VarIds_[i].getName();
      auto Array = createArray(TheFunction, VarName, VarType, SizeExpr);
      auto Alloca = createVariable(TheFunction, VarName, VarPointerType);
      Builder_.CreateStore(Array.Data, Alloca);
      ArrayAllocas.emplace_back( Alloca ); 
    }

    copyArrays(TheFunction, Alloca, ArrayAllocas, SizeExpr );

  }
  
  //---------------------------------------------------------------------------
  // Scalar Initializer
  else {
  
    // Emit initializer first
    auto InitVal = runExprVisitor(*e.InitExpr_);

    // create a size expr
    Value* SizeExpr = nullptr;
    if (e.SizeExpr_)
      SizeExpr = runExprVisitor(*e.SizeExpr_);
    // otherwise scalar
    else
      SizeExpr = llvmInteger(TheContext_, 1);
 
    // Register all variables and emit their initializer.
    std::vector<AllocaInst*> ArrayAllocas;
    for (const auto & VarId : e.VarIds_) {
      const auto & VarName = VarId.getName();
      auto Array = createArray(TheFunction, VarName, VarType, SizeExpr);
      auto Alloca = createVariable(TheFunction, VarName, VarPointerType);
      ArrayAllocas.emplace_back(Alloca);
      Builder_.CreateStore(Array.Data, Alloca);
    }

    initArrays(TheFunction, ArrayAllocas, InitVal, SizeExpr);

    ValueResult_ = InitVal;

  } // else
  //---------------------------------------------------------------------------

  emitLocation(&e);
}

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function.
//==============================================================================
void CodeGen::dispatch(PrototypeAST &e) {

  unsigned NumArgs = e.ArgIds_.size();

  std::vector<Type *> ArgTypes;
  ArgTypes.reserve(NumArgs);

  for (unsigned i=0; i<NumArgs; ++i) {
    auto VarType = getLLVMType(e.ArgTypeIds_[i]);
    if (e.ArgIsArray_[i]) VarType = PointerType::get(VarType, 0);
    ArgTypes.emplace_back(VarType);
  }
  
  Type* ReturnType = VoidType_;
  if (e.ReturnType_) ReturnType = getLLVMType(e.ReturnType_);
  FunctionType *FT = FunctionType::get(ReturnType, ArgTypes, false);

  Function *F = Function::Create(FT, Function::ExternalLinkage, e.Id_.getName(), &getModule());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args()) Arg.setName(e.ArgIds_[Idx++].getName());

  FunctionResult_ = F;

}

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
void CodeGen::dispatch(FunctionAST& e)
{
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto & P = insertFunction( std::move(e.ProtoExpr_) );
  auto TheFunction = getFunction(P.getName());

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", TheFunction);
  Builder_.SetInsertPoint(BB);

  // Create a subprogram DIE for this function.
  auto Unit = createFile();
  unsigned LineNo = P.getLine();
  unsigned ScopeLine = LineNo;
  DISubprogram *SP = createSubprogram( LineNo, ScopeLine, P.getName(),
      TheFunction->arg_size(), Unit);
  if (SP)
    TheFunction->setSubprogram(SP);

  // Push the current scope.
  pushLexicalBlock(SP);

  // Unset the location for the prologue emission (leading instructions with no
  // location in a function are considered part of the prologue and the debugger
  // will run past them when breaking on a function)
  emitLocation(nullptr);

  // Record the function arguments in the NamedValues map.
  VariableTable_.clear();
  ArrayTable_.clear();

  unsigned ArgIdx = 0;
  for (auto &Arg : TheFunction->args()) {

    // get arg type
    auto ArgType = P.ArgTypes_[ArgIdx];
    // the llvm variable type
    auto LLType = getLLVMType(ArgType);
    if (ArgType.isArray()) LLType = PointerType::get(LLType, 0);

    // Create an alloca for this variable.
    AllocaInst *Alloca = createVariable(TheFunction, Arg.getName(), LLType);
    
    // Create a debug descriptor for the variable.
    createVariable( SP, Arg.getName(), ++ArgIdx, Unit, LineNo, Alloca);

    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
  
  }
 

  for ( auto & stmt : e.BodyExprs_ )
  {
    emitLocation(stmt.get());
    runExprVisitor(*stmt);
  }

  // garbage collection
  destroyArrays();
    
  // Finish off the function.
  if ( e.ReturnExpr_ )  {
    auto RetVal = runExprVisitor(*e.ReturnExpr_);
    if (RetVal->getType()->isVoidTy() )
      Builder_.CreateRetVoid();
    else
      Builder_.CreateRet(RetVal);
  }
  else {  
    Builder_.CreateRetVoid();
  }

  // Validate the generated code, checking for consistency.
  verifyFunction(*TheFunction);
    
  FunctionResult_ = TheFunction;
}



} // namespace
