#include "llvm_utils.hpp"

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

} // namespace
