#include "llvm_utils.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

namespace utils {

//============================================================================
std::vector<Type*>  llvmTypes(const std::vector<Value*> & Vals)
{
  std::vector<Type*> Types;
  Types.reserve(Vals.size());
  for (const auto & V : Vals) Types.emplace_back(V->getType());
  return Types;
}

//============================================================================
Constant* llvmString(
    LLVMContext & TheContext,
    Module &TheModule,
    const std::string & Str)
{
  auto ConstantArray = ConstantDataArray::getString(TheContext, Str);
  auto GVStr = new GlobalVariable(
      TheModule,
      ConstantArray->getType(),
      true,
      GlobalValue::InternalLinkage,
      ConstantArray);
  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
  std::vector<Value*> IndicesC = {ZeroC, ZeroC};
  auto StrV = ConstantExpr::getGetElementPtr(
      nullptr, GVStr, IndicesC, true);
  return StrV;
}

//============================================================================
Constant* llvmArray(
    LLVMContext & TheContext,
    Module &TheModule,
    const std::vector<Constant*> & ValsC,
    const std::vector<Constant*> & GEPIndices)
{
  auto ValT = ValsC.front()->getType();
  auto NumArgs = ValsC.size();
  auto ArrayT = ArrayType::get(ValT, NumArgs);
  auto ArrayC = ConstantArray::get(ArrayT, ValsC);
  auto GVStr = new GlobalVariable(
      TheModule,
      ArrayT,
      true,
      GlobalValue::InternalLinkage,
      ArrayC);
  if (GEPIndices.empty()) {
    auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
    return ConstantExpr::getGetElementPtr(nullptr, GVStr, ZeroC, true);
  }
  else {
    return ConstantExpr::getGetElementPtr(nullptr, GVStr, GEPIndices, true);
  }
}

//==============================================================================
void initializeAllTargets() {
  // Initialize the target registry etc.
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();
}

//==============================================================================
const Target* findTarget(const std::string & Type)
{
  for(auto & Tgt : TargetRegistry::targets()) {
    if (Type == Tgt.getName()) {
      return &Tgt;
    }
  }
  return nullptr;
}

//==============================================================================
void startLLVM() {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  
}

//==============================================================================
void copyComdat(GlobalObject *Dst, const GlobalObject *Src) {
  const Comdat *SC = Src->getComdat();
  if (!SC) return;
  Comdat *DC = Dst->getParent()->getOrInsertComdat(SC->getName());
  DC->setSelectionKind(SC->getSelectionKind());
  Dst->setComdat(DC);
}

//==============================================================================
void insertModule(
    Module &SrcM,
    Module & TgtM)
{
  ValueToValueMapTy VMap;
  auto ShouldCloneDefinition = [](const GlobalValue *GV) { return true; };

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (auto I = SrcM.global_begin(), E = SrcM.global_end(); I != E; ++I)
  {
    auto GV = new GlobalVariable(
        TgtM,
        I->getValueType(),
        I->isConstant(), I->getLinkage(),
        (Constant*) nullptr, I->getName(),
        (GlobalVariable*) nullptr,
        I->getThreadLocalMode(),
        I->getType()->getAddressSpace());
    
    GV->copyAttributesFrom(&*I);
    VMap[&*I] = GV;
  }
 
  // Loop over the functions in the module, making external functions as before
  for (const auto &I : SrcM) {
    auto NF = Function::Create(
        cast<FunctionType>(I.getValueType()),
        I.getLinkage(),
        I.getAddressSpace(),
        I.getName(),
        &TgtM);
    NF->copyAttributesFrom(&I);
    VMap[&I] = NF;
  }

  // Loop over the aliases in the module
  for (auto I = SrcM.alias_begin(), E = SrcM.alias_end(); I != E; ++I) 
  {
    if (!ShouldCloneDefinition(&*I)) {
      // An alias cannot act as an external reference, so we need to create
      // either a function or a global variable depending on the value type.
      // FIXME: Once pointee types are gone we can probably pick one or the
      // other.
      GlobalValue *GV;
      if (I->getValueType()->isFunctionTy())
        GV = Function::Create(
            cast<FunctionType>(I->getValueType()),
            GlobalValue::ExternalLinkage,
            I->getAddressSpace(),
            I->getName(),
            &TgtM);
      else
        GV = new GlobalVariable(
            TgtM,
            I->getValueType(),
            false,
            GlobalValue::ExternalLinkage,
            nullptr,
            I->getName(),
            nullptr,
            I->getThreadLocalMode(),
            I->getType()->getAddressSpace());
      VMap[&*I] = GV;
      // We do not copy attributes (mainly because copying between different
      // kinds of globals is forbidden), but this is generally not required for
      // correctness.
      continue;
    }
    auto *GA = GlobalAlias::create(
        I->getValueType(),
        I->getType()->getPointerAddressSpace(),
        I->getLinkage(),
        I->getName(),
        &TgtM);
    GA->copyAttributesFrom(&*I);
    VMap[&*I] = GA;
  }
 
  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (auto I = SrcM.global_begin(), E = SrcM.global_end(); I != E; ++I) 
  {
    if (I->isDeclaration()) continue;
 
    auto GV = cast<GlobalVariable>(VMap[&*I]);
    if (!ShouldCloneDefinition(&*I)) {
      // Skip after setting the correct linkage for an external reference.
      GV->setLinkage(GlobalValue::ExternalLinkage);
      continue;
    }
    if (I->hasInitializer())
      GV->setInitializer(MapValue(I->getInitializer(), VMap));
 
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    I->getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(
          MD.first,
          *MapMetadata(MD.second, VMap, RF_MoveDistinctMDs));
 
    copyComdat(GV, &*I);
  }
 
  // Similarly, copy over function bodies now...
  //
  for (const auto &I : SrcM) {
    if (I.isDeclaration()) continue;
 
    auto F = cast<Function>(VMap[&I]);
    if (!ShouldCloneDefinition(&I)) {
      // Skip after setting the correct linkage for an external reference.
      F->setLinkage(GlobalValue::ExternalLinkage);
      // Personality function is not valid on a declaration.
      F->setPersonalityFn(nullptr);
      continue;
    }
 
    auto DestI = F->arg_begin();
    for (auto J = I.arg_begin(); J != I.arg_end(); ++J) {
      DestI->setName(J->getName());
      VMap[&*J] = &*DestI++;
    }
 
    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, &I, VMap, /*ModuleLevelChanges=*/true, Returns);
 
    if (I.hasPersonalityFn())
      F->setPersonalityFn(MapValue(I.getPersonalityFn(), VMap));
 
    copyComdat(F, &I);
  }

  // And aliases
  for (auto I = SrcM.alias_begin(), E = SrcM.alias_end(); I != E; ++I) 
  {
    // We already dealt with undefined aliases above.
    if (!ShouldCloneDefinition(&*I)) continue;
    auto GA = cast<GlobalAlias>(VMap[&*I]);
    if (const auto C = I->getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }
 
  // And named metadata....
  const auto* LLVM_DBG_CU = SrcM.getNamedMetadata("llvm.dbg.cu");
  for (auto I = SrcM.named_metadata_begin(), E = SrcM.named_metadata_end(); I != E; ++I)
  {
    const auto &NMD = *I;
    auto NewNMD = TgtM.getOrInsertNamedMetadata(NMD.getName());
    if (&NMD == LLVM_DBG_CU) {
      // Do not insert duplicate operands.
      SmallPtrSet<const void*, 8> Visited;
      for (const auto* Operand : NewNMD->operands())
        Visited.insert(Operand);
      for (const auto* Operand : NMD.operands()) {
        auto* MappedOperand = MapMetadata(Operand, VMap);
        if (Visited.insert(MappedOperand).second)
          NewNMD->addOperand(MappedOperand);
      }
    } else
      for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
        NewNMD->addOperand(MapMetadata(NMD.getOperand(i), VMap));
  }

}

//==============================================================================
std::string verifyModule(llvm::Module& M){
  SmallString<SmallVectorLength> SmallStr;
  raw_svector_ostream Stream(SmallStr);
  verifyModule(M, &Stream);
  return Stream.str().str();
}

//==============================================================================
void cloneFunction(Function* F, Module* M)
{
  std::vector<Type*> ArgTypes;
  ValueToValueMapTy VMap;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  for (const Argument &I : F->args())
    if (VMap.count(&I) == 0) // Haven't mapped the argument to anything yet?
      ArgTypes.push_back(I.getType());
 
  // Create a new function type...
  FunctionType *FTy = FunctionType::get(
    F->getFunctionType()->getReturnType(),
    ArgTypes,
    F->getFunctionType()->isVarArg());
 
  // Create the new function...
  Function *NewF = Function::Create(
      FTy,
      F->getLinkage(),
      F->getAddressSpace(),
      F->getName(),
      M);

  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();
  for (const Argument & I : F->args())
    if (VMap.count(&I) == 0) {     // Is this argument preserved?
      DestI->setName(I.getName()); // Copy the name over...
      VMap[&I] = &*DestI++;        // Add mapping to VMap
    }

  bool ModuleLevelChanges = F->getSubprogram() != nullptr;
  SmallVector< ReturnInst *, 8> Returns;
  CloneFunctionInto(NewF, F, VMap, ModuleLevelChanges, Returns);
}

} // namespace
