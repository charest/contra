#include "ast.hpp"
#include "config.hpp"
#include "errors.hpp"

#include "librt/librt.hpp"

#include "llvm/IR/Type.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

using namespace llvm;

namespace contra {

//==============================================================================
// Constructor
//==============================================================================
CodeGen::CodeGen (bool debug = false) :
  Builder_(TheContext_)
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
Function *CodeGen::getFunction(std::string Name, int Line) {

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
    return FI->second->codegen(*this);

  // If no existing prototype exists, return null.
  THROW_SYNTAX_ERROR("'" << Name << "' does not have a valid prototype", Line);

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

} // namespace
