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
Value* llvmString(
    LLVMContext & TheContext,
    Module &TheModule,
    const std::string & Str)
{
  auto ConstantArray = ConstantDataArray::getString(TheContext, Str);
  auto GVStr = new GlobalVariable(TheModule, ConstantArray->getType(), true,
      GlobalValue::InternalLinkage, ConstantArray);
  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
  std::vector<Value*> IndicesC = {ZeroC, ZeroC};
  auto StrV = ConstantExpr::getGetElementPtr(
      nullptr, GVStr, IndicesC, true);
  return StrV;
}

//============================================================================  
IRBuilder<> createBuilder(Function *TheFunction)
{
  auto & Block = TheFunction->getEntryBlock();
  return IRBuilder<>(&Block, Block.begin());
}

//============================================================================  
AllocaInst* createEntryBlockAlloca(
    Function *TheFunction,
    Type* Ty,
    const std::string & Name)
{
  auto TmpB = createBuilder(TheFunction);
  return TmpB.CreateAlloca(Ty, nullptr, Name.c_str());
}

//============================================================================  
Value* getTypeSize(
    IRBuilder<> & Builder,
    Type* ElementType,
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

//============================================================================  
// copy value into alloca if necessary
AllocaInst* getAsAlloca(
    IRBuilder<> &Builder,
    Function* TheFunction,
    Type* ValueT,
    Value* ValueV)
{
  AllocaInst* ValueA = dyn_cast<AllocaInst>(ValueV);
  if (!ValueA) {
    ValueA = createEntryBlockAlloca(TheFunction, ValueT);
    Builder.CreateStore(ValueV, ValueA);
  }
  return ValueA;
}

AllocaInst* getAsAlloca(
    IRBuilder<> &Builder,
    Function* TheFunction,
    Value* ValueV)
{
  AllocaInst* ValueA = dyn_cast<AllocaInst>(ValueV);
  if (!ValueA) {
    auto ValueT = ValueV->getType();
    ValueA = createEntryBlockAlloca(TheFunction, ValueT);
    Builder.CreateStore(ValueV, ValueA);
  }
  return ValueA;
}

//============================================================================  
// Load a value if necessary
Value* getAsValue(
    IRBuilder<> &Builder,
    Type* ValueT,
    Value* ValueV)
{
  AllocaInst* ValueA = dyn_cast<AllocaInst>(ValueV);
  if (ValueA) {
    return Builder.CreateLoad(ValueT, ValueA);
  }
  return ValueV;
}

Value* getAsValue(
    IRBuilder<> &Builder,
    Value* ValueV)
{
  AllocaInst* ValueA = dyn_cast<AllocaInst>(ValueV);
  if (ValueA) {
    auto ValueT = ValueA->getAllocatedType();
    return Builder.CreateLoad(ValueT, ValueA);
  }
  return ValueV;
}

} // namespace
