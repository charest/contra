#include "tasking.hpp"

namespace contra {

using namespace llvm;

//==============================================================================
Type* AbstractTasker::reduceStruct(StructType * StructT, const Module &TheModule) const
{
  auto NumElem = StructT->getNumElements();
  auto ElementTs = StructT->elements();
  if (NumElem == 1) return ElementTs[0];
  auto DL = std::make_unique<DataLayout>(&TheModule);
  auto BitWidth = DL->getTypeAllocSizeInBits(StructT);
  return IntegerType::get(TheContext_, BitWidth);
}

//==============================================================================
Value* AbstractTasker::sanitize(Value* V, const Module &TheModule) const
{
  auto T = V->getType();
  if (auto StrucT = dyn_cast<StructType>(T)) {
    auto TheBlock = Builder_.GetInsertBlock();
    auto NewT = reduceStruct(StrucT, TheModule);
    std::string Str = StrucT->hasName() ? StrucT->getName().str()+".cast" : "casttmp";
    auto Cast = CastInst::Create(CastInst::BitCast, V, NewT, Str, TheBlock);
    return Cast;
  }
  else {
    return V;
  }
}

//==============================================================================
void AbstractTasker::sanitize(std::vector<Value*> & Vs, const Module &TheModule ) const
{ for (auto & V : Vs ) V = sanitize(V, TheModule); }

//==============================================================================
Value* AbstractTasker::load(Value * Alloca, const Module &TheModule,
    std::string Str) const
{
  if (!Str.empty()) Str += ".";
  auto AllocaT = Alloca->getType();
  auto BaseT = AllocaT->getPointerElementType();
  if (auto StructT = dyn_cast<StructType>(BaseT)) {
    auto TheBlock = Builder_.GetInsertBlock();
    auto ReducedT = reduceStruct(StructT, TheModule);
    auto Cast = CastInst::Create(CastInst::BitCast, Alloca,
      ReducedT->getPointerTo(), Str+"alloca.cast", TheBlock);
    return Builder_.CreateLoad(ReducedT, Cast, Str);
  }
  else {
    return Builder_.CreateLoad(BaseT, Alloca, Str);
  }
}

//==============================================================================
void AbstractTasker::store(Value* Val, AllocaInst * Alloca) const
{
  auto BaseT = Alloca->getType()->getPointerElementType();
  if (isa<StructType>(BaseT)) {
    std::vector<Value*> MemberIndices(2);
    MemberIndices[0] = ConstantInt::get(TheContext_, APInt(32, 0, true));
    MemberIndices[1] = ConstantInt::get(TheContext_, APInt(32, 0, true));
    auto GEP = Builder_.CreateGEP( BaseT, Alloca, MemberIndices );
    Builder_.CreateStore(Val, GEP);
  }
  else {
    Builder_.CreateStore(Val, Alloca);
  }
}
  
//==============================================================================
Value* AbstractTasker::start(llvm::Module & TheModule, int Argc, char ** Argv)
{ 
  setStarted();
  return startRuntime(TheModule, Argc, Argv);
};

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name)
{
  auto Id = getNextId();
  auto it = TaskTable_.emplace(Name, TaskInfo{Id});
  return it.first->second;
}

//==============================================================================
TaskInfo & AbstractTasker::insertTask(const std::string & Name, Function* F)
{
  auto TaskName = F->getName();
  auto Id = getNextId();
  auto it = TaskTable_.emplace(Name, TaskInfo{Id, TaskName, F});
  return it.first->second;
}

//==============================================================================
void AbstractTasker::preregisterTasks(llvm::Module & TheModule)
{
  for (const auto & task_pair : TaskTable_ )
    preregisterTask(TheModule, task_pair.first, task_pair.second);
}

//==============================================================================
void AbstractTasker::postregisterTasks(llvm::Module & TheModule)
{
  for (const auto & task_pair : TaskTable_ )
    postregisterTask(TheModule, task_pair.first, task_pair.second);
}
 
} // namespace
