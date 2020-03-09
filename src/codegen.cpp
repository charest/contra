#include "ast.hpp"
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
CodeGen::CodeGen (std::shared_ptr<BinopPrecedence> Precedence,
    bool debug = false) : Builder_(TheContext_), BinopPrecedence_(Precedence)
{

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
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return runFuncVisitor(*FI->second);
  
  return nullptr;
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

//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
ArrayType
CodeGen::createArray(Function *TheFunction, const std::string &VarName,
    Type * PtrType, Value * SizeExpr)
{


  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());

  Function *F; 
  F = TheModule_->getFunction("allocate");
  if (!F) F = librt::RunTimeLib::tryInstall(TheContext_, *TheModule_, "allocate");


  auto ElementType = PtrType->getPointerElementType();
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

  return ArrayType{AllocInst, Cast, SizeExpr};
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
  
    auto CallInst = Builder_.CreateCall(F, Vec, Name+"dealloctmp");
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
// ExprAST - Base expression class.
//==============================================================================
void CodeGen::dispatch(ExprAST& e)
{ e.accept(*this); }


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

  // Look this variable up in the function.
  auto it = NamedValues.find(e.Name_);
  if (it == NamedValues.end()) 
    THROW_NAME_ERROR(e.Name_, e.getLine());
  emitLocation(&e);

  Value* V = it->second;
  
  // Load the value.
  auto Ty = V->getType();
  if (!Ty->isPointerTy()) THROW_CONTRA_ERROR("why are you NOT a pointer");
  Ty = Ty->getPointerElementType();
    
  auto Load = Builder_.CreateLoad(Ty, V, "val."+e.Name_);

  if ( !e.Index_ ) {
    //if (TheCG.NamedArrays.count(Name_))
    //  THROW_SYNTAX_ERROR("Array accesses require explicit indices", getLine());
    ValueResult_ = Load;
  }
  else {
    Ty = Ty->getPointerElementType();
    auto IndexVal = runExprVisitor(*e.Index_);
    auto GEP = Builder_.CreateGEP(Load, IndexVal, e.Name_+".offset");
    ValueResult_ = Builder_.CreateLoad(Ty, GEP, e.Name_+".i");
  }
}

//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
void CodeGen::dispatch(ArrayExprAST &e)
{
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // the llvm variable type
  Type * VarType;
  try {
    VarType = getLLVMType(e.InferredType, TheContext_);
  }
  catch (const ContraError & err) {
    THROW_SYNTAX_ERROR( "Unknown variable type of '" << getVarTypeName(e.InferredType)
        << "' used in array initialization", e.getLine() );
  }
  auto VarPointerType = PointerType::get(VarType, 0);


  std::vector<Value*> InitVals;
  InitVals.reserve(e.Vals_.size());
  for ( auto & E : e.Vals_ ) 
    InitVals.emplace_back( runExprVisitor(*E) );

  Value* SizeExpr = nullptr;
  if (e.Size_) {
    SizeExpr = runExprVisitor(*e.Size_);
    if (e.Vals_.size() != 1 )
      THROW_SYNTAX_ERROR("Only one value expected in [Val; N] syntax", e.getLine());
  }
  else {
    SizeExpr = llvmInteger(TheContext_, e.Vals_.size());
  }

  auto Array = createArray(TheFunction, "__tmp", VarPointerType, SizeExpr );
  auto Alloca = createEntryBlockAlloca(TheFunction, "__tmp", VarPointerType);
  Builder_.CreateStore(Array.Data, Alloca);

  if (e.Size_) 
    initArrays(TheFunction, {Alloca}, InitVals[0], SizeExpr);
  else
    initArray(TheFunction, Alloca, InitVals);

  TempArrays[Array.Alloca] = Array;

  ValueResult_ = Array.Alloca;
}

//==============================================================================
// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
void CodeGen::dispatch(BinaryExprAST& e) {
  emitLocation(&e);
  
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (e.Op_ == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    auto LHSE = std::dynamic_pointer_cast<VariableExprAST>(e.LHS_);
    if (!LHSE)
      THROW_SYNTAX_ERROR("destination of '=' must be a variable", LHSE->getLine());
    // Codegen the RHS.
    auto Val = runExprVisitor(*e.RHS_);

    // Look up the name.
    const auto & VarName = LHSE->getName();
    Value *Variable = NamedValues[VarName];
    if (!Variable)
      THROW_NAME_ERROR(VarName, LHSE->getLine());

    //if (TheCG.NamedArrays.count(VarName)) {
    if (Variable->getType()->getPointerElementType()->isPointerTy()) {
      //if (!LHSE->isArray())
      //  THROW_SYNTAX_ERROR("Arrays must be indexed using '[i]'", LHSE->getLine());
      auto Ty = Variable->getType()->getPointerElementType();
      auto Load = Builder_.CreateLoad(Ty, Variable, "ptr."+VarName);
      auto IndexVal = runExprVisitor(*LHSE->getIndex());
      auto GEP = Builder_.CreateGEP(Load, IndexVal, VarName+"aoffset");
      Builder_.CreateStore(Val, GEP);
    }
    else {
      Builder_.CreateStore(Val, Variable);
    }
    ValueResult_ = Val;
    return;
  }

  Value *L = runExprVisitor(*e.LHS_);
  Value *R = runExprVisitor(*e.RHS_);

  auto l_is_real = L->getType()->isFloatingPointTy();
  auto r_is_real = R->getType()->isFloatingPointTy();
  bool is_real =  (l_is_real || r_is_real);

  if (is_real) {
    auto TheBlock = Builder_.GetInsertBlock();
    if (!l_is_real) {
      auto cast = CastInst::Create(Instruction::SIToFP, L,
          llvmRealType(TheContext_), "castl", TheBlock);
      L = cast;
    }
    else if (!r_is_real) {
      auto cast = CastInst::Create(Instruction::SIToFP, R,
          llvmRealType(TheContext_), "castr", TheBlock);
      R = cast;
    }
  }

  if (is_real) {
    switch (e.Op_) {
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
    switch (e.Op_) {
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
  auto F = getFunction(std::string("binary") + e.Op_);
  if (!F) THROW_CONTRA_ERROR("binary operator not found!");

  Value *Ops[] = { L, R };
  ValueResult_ = Builder_.CreateCall(F, Ops, "binop");
}

//==============================================================================
// CallExprAST - Expression class for function calls.
//==============================================================================
void CodeGen::dispatch(CallExprAST &e) {
  emitLocation(&e);
  
  // special cases
  auto TheBlock = Builder_.GetInsertBlock();
  if (e.Callee_ == Tokens::getName(tok_int)) {
    auto A = runExprVisitor(*e.Args_[0]);
    if (A->getType()->isFloatingPointTy()) {
      auto cast = CastInst::Create(Instruction::FPToSI, A,
          llvmIntegerType(TheContext_), "cast", TheBlock);
      ValueResult_ = cast;
      return;
    }
    else {
      ValueResult_ = A;
      return;
    }
  }
  else if  (e.Callee_ == Tokens::getName(tok_real)) {
    auto A = runExprVisitor(*e.Args_[0]);
    if (A->getType()->isIntegerTy()) {
      auto cast = CastInst::Create(Instruction::SIToFP, A,
          llvmRealType(TheContext_), "cast", TheBlock);
      ValueResult_ = cast;
      return;
    }
    else {
      ValueResult_ = A;
      return;
    }
  }


  // Look up the name in the global module table.
  auto CalleeF = getFunction(e.Callee_);
  if (!CalleeF)
    THROW_NAME_ERROR(e.Callee_, e.getLine());

  // If argument mismatch error.
  if (CalleeF->arg_size() != e.Args_.size() && !CalleeF->isVarArg()) {
    THROW_SYNTAX_ERROR(
        "Incorrect number of arguments, expected " << CalleeF->arg_size() 
        << " but got " << e.Args_.size() << Formatter::to_str, e.getLine() );
  }

  auto FunType = CalleeF->getFunctionType();
  auto NumFixedArgs = FunType->getNumParams();

  std::vector<Value *> ArgsV;
  for (unsigned i = 0; i<e.Args_.size(); ++i) {
    // what is the arg type
    auto A = runExprVisitor(*e.Args_[i]);
    if (i < NumFixedArgs) {
      auto TheBlock = Builder_.GetInsertBlock();
      if (FunType->getParamType(i)->isFloatingPointTy() && A->getType()->isIntegerTy()) {
        auto cast = CastInst::Create(Instruction::SIToFP, A,
            llvmRealType(TheContext_), "cast", TheBlock);
        A = cast;
      }
      else if (FunType->getParamType(i)->isIntegerTy() && A->getType()->isFloatingPointTy()) {
        auto cast = CastInst::Create(Instruction::FPToSI, A,
            llvmIntegerType(TheContext_), "cast", TheBlock);
        A = cast;
      }
    }
    ArgsV.push_back(A);
  }

  ValueResult_ = Builder_.CreateCall(CalleeF, ArgsV, "calltmp");
}

//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
void CodeGen::dispatch(IfExprAST & e) {
  emitLocation(&e);
  
  if ( e.Then_.empty() && e.Else_.empty() ) {
    ValueResult_ = Constant::getNullValue(llvmIntegerType(TheContext_));
    return;
  }
  else if (e.Then_.empty())
    THROW_SYNTAX_ERROR( "Can't have else with no if!", e.getLine() );


  Value *CondV = runExprVisitor(*e.Cond_);

  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(TheContext_, "then", TheFunction);
  BasicBlock *ElseBB = e.Else_.empty() ? nullptr : BasicBlock::Create(TheContext_, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheContext_, "ifcont");

  if (ElseBB)
    Builder_.CreateCondBr(CondV, ThenBB, ElseBB);
  else
    Builder_.CreateCondBr(CondV, ThenBB, MergeBB);

  // Emit then value.
  Builder_.SetInsertPoint(ThenBB);

  for ( auto & stmt : e.Then_ ) runExprVisitor(*stmt);

  // get first non phi instruction
  auto ThenV = ThenBB->getFirstNonPHI();

  Builder_.CreateBr(MergeBB);

  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder_.GetInsertBlock();

  if (ElseBB) {

    // Emit else block.
    TheFunction->getBasicBlockList().push_back(ElseBB);
    Builder_.SetInsertPoint(ElseBB);

    for ( auto & stmt : e.Else_ ) runExprVisitor(*stmt);

    // get first non phi
    auto ElseV = ElseBB->getFirstNonPHI();

    Builder_.CreateBr(MergeBB);
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = Builder_.GetInsertBlock();

  } // else

  // Emit merge block.
  TheFunction->getBasicBlockList().push_back(MergeBB);
  Builder_.SetInsertPoint(MergeBB);
  //ElseV->getType()->print(outs());
  //outs() << "\n";
  //PHINode *PN = Builder.CreatePHI(ThenV->getType(), 2, "iftmp");

  //if (ThenV) PN->addIncoming(ThenV, ThenBB);
  //if (ElseV) PN->addIncoming(ElseV, ElseBB);
  //return PN;
  
  // for expr always returns 0.
  ValueResult_ = Constant::getNullValue(llvmIntegerType(TheContext_));
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
void CodeGen::dispatch(ForExprAST& e) {
  
  auto TheFunction = Builder_.GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  auto LLType = llvmIntegerType(TheContext_);
  AllocaInst *Alloca = createEntryBlockAlloca(TheFunction, e.VarName_, LLType);
  
  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  AllocaInst *OldVal = NamedValues[e.VarName_];
  NamedValues[e.VarName_] = Alloca;
  emitLocation(&e);

  // Emit the start code first, without 'variable' in scope.
  auto StartVal = runExprVisitor(*e.Start_);
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
  auto EndCond = runExprVisitor(*e.End_);
  if (EndCond->getType()->isFloatingPointTy())
    THROW_IMPLEMENTED_ERROR("Cast required for end condition");
 
  if (e.Loop_ == ForExprAST::LoopType::Until) {
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
  for ( auto & stmt : e.Body_ ) runExprVisitor(*stmt);

  // Insert unconditional branch to increment.
  Builder_.CreateBr(IncrBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  Builder_.SetInsertPoint(IncrBB);
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (e.Step_) {
    StepVal = runExprVisitor(*e.Step_);
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

  // Restore the unshadowed variable.
  if (OldVal)
    NamedValues[e.VarName_] = OldVal;
  else
    NamedValues.erase(e.VarName_);

  // for expr always returns 0.
  ValueResult_ = Constant::getNullValue(LLType);
}
  
//==============================================================================
// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
void CodeGen::dispatch(UnaryExprAST & e) {
  
  auto OperandV = runExprVisitor(*e.Operand_); 
  
  if (OperandV->getType()->isFloatingPointTy()) {
  
    switch (e.Opcode_) {
    case tok_sub:
      ValueResult_ = Builder_.CreateFNeg(OperandV, "negtmp");
      return;
    default:
      THROW_SYNTAX_ERROR( "Uknown unary operator '" << static_cast<char>(e.Opcode_)
          << "'", e.getLine() );
    }

  }
  else {
    switch (e.Opcode_) {
    case tok_sub:
      ValueResult_ = Builder_.CreateNeg(OperandV, "negtmp");
      return;
    default:
      THROW_SYNTAX_ERROR( "Uknown unary operator '" << static_cast<char>(e.Opcode_)
          << "'", e.getLine() );
    }
  }

  auto F = getFunction(std::string("unary") + e.Opcode_);
  if (!F)
    THROW_SYNTAX_ERROR("Unknown unary operator", e.getLine());

  emitLocation(&e);
  ValueResult_ = Builder_.CreateCall(F, OperandV, "unop");
}

//==============================================================================
// VarExprAST - Expression class for var/in
//==============================================================================
void CodeGen::dispatch(VarExprAST & e) {

  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.

  // Emit initializer first
  auto InitVal = runExprVisitor(*e.Init_);
  auto IType = InitVal->getType();

  // the llvm variable type
  Type * VarType;
  try {
    VarType = getLLVMType(e.VarType_, TheContext_);
  }
  catch (const ContraError & err) {
    THROW_SYNTAX_ERROR( "Unknown variable type of '" << getVarTypeName(e.VarType_)
        << "' for variables '" << e.VarNames_ << "'", e.getLine() );
  }


  // Register all variables and emit their initializer.
  for (const auto & VarName : e.VarNames_) {
    
    // cast init value if necessary
    auto TheBlock = Builder_.GetInsertBlock();
    if (e.VarType_ == VarTypes::Real && !InitVal->getType()->isFloatingPointTy()) {
      auto cast = CastInst::Create(Instruction::SIToFP, InitVal,
          llvmRealType(TheContext_), "cast", TheBlock);
      InitVal = cast;
    }
    else if (e.VarType_ == VarTypes::Int && !InitVal->getType()->isIntegerTy()) {
      auto cast = CastInst::Create(Instruction::FPToSI, InitVal,
          llvmIntegerType(TheContext_), "cast", TheBlock);
      InitVal = cast;
    }

    auto Alloca = createEntryBlockAlloca(TheFunction, VarName, VarType);
    Builder_.CreateStore(InitVal, Alloca);
  
    // Remember this binding.
    NamedValues[VarName] = Alloca;
  }


  emitLocation(&e);

#if 0
  // Codegen the body, now that all vars are in scope.
  Value *BodyVal = Body->codegen(TheCG);
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
#endif

  ValueResult_ = InitVal;
}

//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
void CodeGen::dispatch(ArrayVarExprAST &e) {
  auto TheFunction = Builder_.GetInsertBlock()->getParent();
  
  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.

  Value* ReturnInit = nullptr;
  
  // the llvm variable type
  Type * VarType;
  try {
    VarType = getLLVMType(e.VarType_, TheContext_);
  }
  catch (const ContraError & err) {
    THROW_SYNTAX_ERROR( "Unknown variable type of '" << getVarTypeName(e.VarType_)
        << "' for variables '" << e.VarNames_ << "'", e.getLine() );
  }
  auto VarPointerType = PointerType::get(VarType, 0);

  //---------------------------------------------------------------------------
  // Array already on right hand side
  auto ArrayAST = std::dynamic_pointer_cast<ArrayExprAST>(e.Init_);
  if (ArrayAST) {

    // transfer to first
    const auto & VarName = e.VarNames_[0];
    auto ArrayAlloca = static_cast<AllocaInst*>(runExprVisitor(*ArrayAST));
    auto Array = TempArrays[ArrayAlloca];
    TempArrays.erase(ArrayAlloca);
    auto SizeExpr = Array.Size;
    ReturnInit = Array.Data;
    auto FirstAlloca = createEntryBlockAlloca(TheFunction, VarName,
        VarPointerType);
    Builder_.CreateStore(Array.Data, FirstAlloca);
    NamedValues[VarName] = FirstAlloca;
    NamedArrays[VarName] = ArrayAlloca;
  
    // more complicated
    if (e.VarNames_.size() > 1) {
    
      std::vector<AllocaInst*> ArrayAllocas;
      ArrayAllocas.reserve(e.VarNames_.size());

      // Register all variables and emit their initializer.
      for (int i=1; i<e.VarNames_.size(); ++i) {
        const auto & VarName = e.VarNames_[i];
        auto Array = createArray(TheFunction, VarName, VarPointerType, SizeExpr);
        auto Alloca = createEntryBlockAlloca(TheFunction, VarName, VarPointerType);
        Builder_.CreateStore(Array.Data, Alloca);
        NamedValues[VarName] = Alloca;
        NamedArrays[VarName] = Array.Alloca;
        ArrayAllocas.emplace_back( Alloca ); 
      }
      copyArrays(TheFunction, FirstAlloca, ArrayAllocas, SizeExpr );
    }

  }
  
  //---------------------------------------------------------------------------
  // Scalar Initializer
  else {
  
    // Emit initializer first
    auto InitVal = runExprVisitor(*e.Init_);

    // create a size expr
    auto IType = InitVal->getType();
    Value * SizeExpr = nullptr;

    if (e.Size_) {
      SizeExpr = runExprVisitor(*e.Size_);
    }
    else if (IType->isSingleValueType()) {
      SizeExpr = llvmInteger(TheContext_, 1);
    }
    else {
      THROW_SYNTAX_ERROR("Unknown array initialization", e.getLine()); 
    }
 
    std::vector<AllocaInst*> ArrayAllocas;

    // Register all variables and emit their initializer.
    for (const auto & VarName : e.VarNames_) {
      
      // cast init value if necessary
      auto TheBlock = Builder_.GetInsertBlock();
      if (e.VarType_ == VarTypes::Real && !InitVal->getType()->isFloatingPointTy()) {
        auto cast = CastInst::Create(Instruction::SIToFP, InitVal,
            llvmRealType(TheContext_), "cast", TheBlock);
        InitVal = cast;
      }
      else if (e.VarType_ == VarTypes::Int && !InitVal->getType()->isIntegerTy()) {
        auto cast = CastInst::Create(Instruction::FPToSI, InitVal,
            llvmIntegerType(TheContext_), "cast", TheBlock);
        InitVal = cast;
      }

      AllocaInst* Alloca;

      // create array of var
      auto Array = createArray(TheFunction, VarName, VarPointerType, SizeExpr);
  
      NamedArrays[VarName] = Array.Alloca;

      Alloca = createEntryBlockAlloca(TheFunction, VarName, VarPointerType);
      ArrayAllocas.emplace_back(Alloca);

      Builder_.CreateStore(Array.Data, Alloca);
    
      // Remember this binding.
      NamedValues[VarName] = Alloca;
    }

    initArrays(TheFunction, ArrayAllocas, InitVal, SizeExpr);

    ReturnInit = InitVal;

  } // else
  //---------------------------------------------------------------------------

  emitLocation(&e);


  ValueResult_ = ReturnInit;
}

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function.
//==============================================================================
void CodeGen::dispatch(PrototypeAST &e) {

  std::vector<Type *> ArgTypes;
  ArgTypes.reserve(e.Args_.size());

  for ( const auto & A : e.Args_ ) {
    auto VarSymbol = A.second;
    auto VarType = VarSymbol.getType();
    Type * LLType;
    try {
      LLType = getLLVMType(VarType, TheContext_);
    }
    catch (const ContraError & err) {
      THROW_SYNTAX_ERROR( "Unknown argument type of '" << getVarTypeName(VarType)
          << "' in prototype for function '" << e.Name_ << "'", e.Line_ );
    }
    
    if (VarSymbol.isArray())
      LLType = PointerType::get(LLType, 0);

    ArgTypes.emplace_back(LLType);
  }
  
  Type * ReturnType;
  try {
    ReturnType = getLLVMType(e.Return_, TheContext_);
  }
  catch (const ContraError & err) {
    THROW_SYNTAX_ERROR( "Unknown return type of '" << getVarTypeName(e.Return_)
        << "' in prototype for function '" << e.Name_ << "'", e.Line_ );
  }

  FunctionType *FT = FunctionType::get(ReturnType, ArgTypes, false);

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, e.Name_, &getModule());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(e.Args_[Idx++].first);

  FunctionResult_ = F;
}

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
void CodeGen::dispatch(FunctionAST& e)
{
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *e.Proto_;
  FunctionProtos[e.Proto_->getName()] = std::move(e.Proto_);
  auto TheFunction = getFunction(P.getName());
  if (!TheFunction)
    THROW_SYNTAX_ERROR("'" << P.getName() << "' does not have a valid prototype",
        P.getLine());

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence_->operator[](P.getOperatorName()) = P.getBinaryPrecedence();

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
  NamedValues.clear();
  NamedArrays.clear();
  unsigned ArgIdx = 0;
  for (auto &Arg : TheFunction->args()) {

    // get arg type
    const auto & ArgSymbol = P.getArgSymbol(ArgIdx);
    auto ArgType = ArgSymbol.getType();
  
    // the llvm variable type
    Type * LLType;
    try {
      LLType = getLLVMType(ArgType, TheContext_);
    }
    catch (const ContraError & err) {
      THROW_SYNTAX_ERROR( "Unknown variable type of '" << getVarTypeName(ArgType)
          << "' used in function prototype for '" << P.getName() << "'",
          P.getLine() );
    }
    if (ArgSymbol.isArray()) 
      LLType = PointerType::get(LLType, 0);

    // Create an alloca for this variable.
    AllocaInst *Alloca = createEntryBlockAlloca(TheFunction, Arg.getName(), LLType);
    
    // Create a debug descriptor for the variable.
    createVariable( SP, Arg.getName(), ++ArgIdx, Unit, LineNo, Alloca);

    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);

    // Add arguments to variable symbol table.
    NamedValues[Arg.getName()] = Alloca;
  }
 

  for ( auto & stmt : e.Body_ )
  {
    emitLocation(stmt.get());
    runExprVisitor(*stmt);
  }

  // garbage collection
  destroyArrays();
    
  // Finish off the function.
  if ( e.Return_ ) {
    auto RetVal = runExprVisitor(*e.Return_);
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

#if 0

  if (Value *RetVal = Body->codegen(TheCG)) {
    // Pop off the lexical block for the function.
    popLexicalBlock();

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    // Run the optimizer on the function.
    TheFPM->run(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();
  
  if (P.isBinaryOp())
    TheParser.BinopPrecedence.erase(Proto->getOperatorName());

  // Pop off the lexical block for the function since we added it
  // unconditionally.
  popLexicalBlock();
#endif
}



} // namespace
