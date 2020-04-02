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
Value* llvmString(LLVMContext & TheContext,
    Module &TheModule, const std::string & Str)
{
  auto ConstantArray = ConstantDataArray::getString(TheContext, Str);
  auto GVStr = new GlobalVariable(TheModule, ConstantArray->getType(), true,
      GlobalValue::InternalLinkage, ConstantArray);
  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
  auto StrV = ConstantExpr::getGetElementPtr(
      IntegerType::getInt8Ty(TheContext), GVStr, ZeroC, true);
  return StrV;
}

//============================================================================  
IRBuilder<> createBuilder(Function *TheFunction)
{
  auto & Block = TheFunction->getEntryBlock();
  return IRBuilder<>(&Block, Block.begin());
}

//============================================================================  
AllocaInst* createEntryBlockAlloca(Function *TheFunction,
  Type* Ty, const std::string & Name)
{
  auto TmpB = createBuilder(TheFunction);
  return TmpB.CreateAlloca(Ty, nullptr, Name.c_str());
}

//============================================================================  
Value* getTypeSize(IRBuilder<> & Builder, Type* ElementType,
    Type* ResultType )
{
  using namespace llvm;
  auto & TheContext = Builder.getContext();
  auto TheBlock = Builder.GetInsertBlock();
  auto PtrType = ElementType->getPointerTo();
  auto Index = ConstantInt::get(TheContext, APInt(32, 1, true));
  auto Null = Constant::getNullValue(PtrType);
  auto SizeGEP = Builder.CreateGEP(ElementType, Null, Index, "size");
  auto DataSize = CastInst::Create(Instruction::PtrToInt, SizeGEP,
          ResultType, "sizei", TheBlock);
  return DataSize;
}

} // namespace
