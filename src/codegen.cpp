#include "ast.hpp"

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
Function * CodeGen::getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen(*this);

  // If no existing prototype exists, return null.
  return nullptr;
}

//==============================================================================
/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
//==============================================================================
AllocaInst *CodeGen::createEntryBlockAlloca(Function *TheFunction,
    const std::string &VarName)
{
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getDoubleTy(TheContext), nullptr, VarName.c_str());
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
