#include "llvm_utils.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

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
TargetMachine* createTargetMachine(const std::string & Type)
{
  const Target* MyTgt = nullptr; 
  for(auto & Tgt : TargetRegistry::targets()) {
    if (Type == Tgt.getName()) {
      MyTgt = &Tgt;
      break;
    }
  }

  TargetMachine* Machine = nullptr;

  if ( MyTgt ) {
    if (Type == "nvptx64") {
    }
  }

  return Machine;
}

//==============================================================================
void startLLVM() {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  
}

} // namespace
