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
  Builder(TheContext)
{

  initializeModuleAndPassManager();

  if (debug) {

    // Add the current debug info version into the module.
    TheModule->addModuleFlag(Module::Warning, "Debug Info Version",
                             DEBUG_METADATA_VERSION);

    // Darwin only supports dwarf2.
    if (Triple(sys::getProcessTriple()).isOSDarwin())
      TheModule->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);

    DBuilder = std::make_unique<DIBuilder>(*TheModule);
    KSDbgInfo.TheCU = DBuilder->createCompileUnit(
      dwarf::DW_LANG_C, DBuilder->createFile("fib.ks", "."),
      "Kaleidoscope Compiler", 0, "", 0);
  }
}

//==============================================================================
// Top-Level parsing and JIT Driver
//==============================================================================
void CodeGen::initializeModuleAndPassManager() {
  initializeModule();
  if (!isDebug())
    initializePassManager();
}


//==============================================================================
void CodeGen::initializeModule() {

  // Open a new module.
  TheModule = std::make_unique<Module>("my cool jit", TheContext);
  TheModule->setDataLayout(TheJIT.getTargetMachine().createDataLayout());

}


//==============================================================================
void CodeGen::initializePassManager() {

  // Create a new pass manager attached to it.
  TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Promote allocas to registers.
  TheFPM->add(createPromoteMemoryToRegisterPass());
  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());
  TheFPM->doInitialization();
}

//==============================================================================
// Get the function
//==============================================================================
Function *CodeGen::getFunction(std::string Name, int Line) {

  // First, see if the function has already been added to the current module.
  if (auto F = TheModule->getFunction(Name))
    return F;
  
  // see if this is an available intrinsic, try installing it first
  if (auto F = librt::RunTimeLib::tryInstall(TheContext, *TheModule, Name))
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
CodeGen::ArrayType
CodeGen::createArray(Function *TheFunction, const std::string &VarName,
    Type * PtrType, Value * SizeExpr)
{

  //----------------------------------------------------------------------------
  // Create Array

  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());

  Function *F; 
  F = TheModule->getFunction("allocate");
  if (!F) F = librt::RunTimeLib::tryInstall(TheContext, *TheModule, "allocate");


  auto ElementType = PtrType->getPointerElementType();
  auto TheBlock = Builder.GetInsertBlock();
  auto Index = ConstantInt::get(TheContext, APInt(32, 1, true));
  auto Null = Constant::getNullValue(PtrType);
  auto SizeGEP = Builder.CreateGEP(ElementType, Null, Index, "size");
  auto DataSize = CastInst::Create(Instruction::PtrToInt, SizeGEP,
          llvmIntegerType(TheContext), "sizei", TheBlock);

  auto TotalSize = Builder.CreateMul(SizeExpr, DataSize, "multmp");

  Value* CallInst = Builder.CreateCall(F, TotalSize, VarName+"vectmp");
  auto ResType = CallInst->getType();
  auto AllocInst = TmpB.CreateAlloca(ResType, 0, VarName+"vec");
  Builder.CreateStore(CallInst, AllocInst);

  std::vector<Value*> MemberIndices(2);
  MemberIndices[0] = ConstantInt::get(TheContext, APInt(32, 0, true));
  MemberIndices[1] = ConstantInt::get(TheContext, APInt(32, 0, true));

  auto GEPInst = Builder.CreateGEP(ResType, AllocInst, MemberIndices,
      VarName+"vec.ptr");
  auto LoadedInst = Builder.CreateLoad(GEPInst->getType()->getPointerElementType(),
      GEPInst, VarName+"vec.val");

  Value* Cast = CastInst::Create(CastInst::BitCast, LoadedInst, PtrType, "casttmp", TheBlock);

  return ArrayType{AllocInst, Cast};
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
  
  auto Alloca = TmpB.CreateAlloca(llvmIntegerType(TheContext), nullptr, "__i");
  Value * StartVal = ConstantInt::get(TheContext, APInt(64, 0, true));
  Builder.CreateStore(StartVal, Alloca);
  
  auto BeforeBB = BasicBlock::Create(TheContext, "beforeinit", TheFunction);
  auto LoopBB =   BasicBlock::Create(TheContext, "init", TheFunction);
  auto AfterBB =  BasicBlock::Create(TheContext, "afterinit", TheFunction);
  Builder.CreateBr(BeforeBB);
  Builder.SetInsertPoint(BeforeBB);
  auto CurVar = Builder.CreateLoad(llvmIntegerType(TheContext), Alloca);
  auto EndCond = Builder.CreateICmpSLT(CurVar, SizeExpr, "initcond");
  Builder.CreateCondBr(EndCond, LoopBB, AfterBB);
  Builder.SetInsertPoint(LoopBB);

  for ( auto i : VarList) {
    auto LoadType = i->getType()->getPointerElementType();
    auto Load = Builder.CreateLoad(LoadType, i, "ptr"); 
    auto GEP = Builder.CreateGEP(Load, CurVar, "offset");
    Builder.CreateStore(InitVal, GEP);
  }

  auto StepVal = ConstantInt::get(TheContext, APInt(64, 1, true));
  auto NextVar = Builder.CreateAdd(CurVar, StepVal, "nextvar");
  Builder.CreateStore(NextVar, Alloca);
  Builder.CreateBr(BeforeBB);
  Builder.SetInsertPoint(AfterBB);
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
  auto TheBlock = Builder.GetInsertBlock();

  auto LoadType = Var->getType()->getPointerElementType();
  auto ValType = LoadType->getPointerElementType();
  auto Load = Builder.CreateLoad(LoadType, Var, "ptr"); 
  
  for (std::size_t i=0; i<NumVals; ++i) {
    auto Index = ConstantInt::get(TheContext, APInt(64, i, true));
    auto GEP = Builder.CreateGEP(Load, Index, "offset");
    auto Init = InitVals[i];
    auto InitType = Init->getType();
    if ( InitType->isFloatingPointTy() && ValType->isIntegerTy() ) {
      auto Cast = CastInst::Create(Instruction::FPToSI, Init,
          llvmIntegerType(TheContext), "cast", TheBlock);
      Init = Cast;
    }
    else if ( InitType->isIntegerTy() && ValType->isFloatingPointTy() ) {
      auto Cast = CastInst::Create(Instruction::SIToFP, Init,
          llvmRealType(TheContext), "cast", TheBlock);
      Init = Cast;
    }
    else if (InitType!=ValType)
      THROW_CONTRA_ERROR("Unknown cast operation");
    Builder.CreateStore(Init, GEP);
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

  auto Alloca = TmpB.CreateAlloca(llvmIntegerType(TheContext), nullptr, "__i");
  Value * StartVal = ConstantInt::get(TheContext, APInt(64, 0, true));
  Builder.CreateStore(StartVal, Alloca);
  
  auto BeforeBB = BasicBlock::Create(TheContext, "beforeinit", TheFunction);
  auto LoopBB =   BasicBlock::Create(TheContext, "init", TheFunction);
  auto AfterBB =  BasicBlock::Create(TheContext, "afterinit", TheFunction);
  Builder.CreateBr(BeforeBB);
  Builder.SetInsertPoint(BeforeBB);
  auto CurVar = Builder.CreateLoad(llvmIntegerType(TheContext), Alloca);
  auto EndCond = Builder.CreateICmpSLT(CurVar, NumElements, "initcond");
  Builder.CreateCondBr(EndCond, LoopBB, AfterBB);
  Builder.SetInsertPoint(LoopBB);
    
  auto PtrType = Src->getType()->getPointerElementType();
  auto ValType = PtrType->getPointerElementType();

  auto SrcLoad = Builder.CreateLoad(PtrType, Src, "srcptr"); 
  auto SrcGEP = Builder.CreateGEP(SrcLoad, CurVar, "srcoffset");
  auto SrcVal = Builder.CreateLoad(ValType, SrcLoad, "srcval");

  for ( auto T : Tgts ) {
    auto TgtLoad = Builder.CreateLoad(PtrType, T, "tgtptr"); 
    auto TgtGEP = Builder.CreateGEP(TgtLoad, CurVar, "tgtoffset");
    Builder.CreateStore(SrcVal, TgtGEP);
  }

  auto StepVal = ConstantInt::get(TheContext, APInt(64, 1, true));
  auto NextVar = Builder.CreateAdd(CurVar, StepVal, "nextvar");
  Builder.CreateStore(NextVar, Alloca);
  Builder.CreateBr(BeforeBB);
  Builder.SetInsertPoint(AfterBB);
}
  

//==============================================================================
// Destroy all arrays
//==============================================================================
void CodeGen::destroyArrays() {
  
  Function *F; 
  F = TheModule->getFunction("deallocate");
  if (!F) F = librt::RunTimeLib::tryInstall(TheContext, *TheModule, "deallocate");
  
  for ( auto & i : NamedArrays )
  {
    const auto & Name = i.first;
    auto & Alloca = i.second;
    auto AllocaT = Alloca->getType()->getPointerElementType();
    auto Vec = Builder.CreateLoad(AllocaT, Alloca, Name+"vec");
  
    auto CallInst = Builder.CreateCall(F, Vec, Name+"dealloctmp");
  }

  NamedArrays.clear();
}


//==============================================================================
// JIT the current module
//==============================================================================
JIT::VModuleKey CodeGen::doJIT()
{
  auto H = TheJIT.addModule(std::move(TheModule));
  initializeModuleAndPassManager();
  return H;
}

//==============================================================================
// Search the JIT for a symbol
//==============================================================================
JIT::JITSymbol CodeGen::findSymbol( const char * Symbol )
{ return TheJIT.findSymbol(Symbol); }

//==============================================================================
// Delete a JITed module
//==============================================================================
void CodeGen::removeJIT( JIT::VModuleKey H )
{ TheJIT.removeModule(H); }

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
        Builder.GetInsertBlock());

    return D;
  }
  else {
    return nullptr;
  }
}

} // namespace
