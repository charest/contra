#include "config.hpp"
#include "errors.hpp"
#include "legion.hpp"

#include "legion/legion_c.h"

#include <vector>

namespace contra {

using namespace llvm;

//==============================================================================
// Create the opaque type
//==============================================================================
Type * createOpaqueType(const std::string & Name, LLVMContext & TheContext)
{
  auto OpaqueType = StructType::create( TheContext, Name );
  auto VoidPointerType = llvmVoidPointerType(TheContext);

  std::vector<Type*> members{ VoidPointerType }; 
  OpaqueType->setBody( members );

  return OpaqueType;
}

//==============================================================================
// Create integer types
//==============================================================================
template<typename T>
constexpr Type * createIdType(LLVMContext & TheContext)
{
  switch (sizeof(T)) {
  case  1: return Type::getInt8Ty(TheContext);
  case  2: return Type::getInt16Ty(TheContext);
  case  4: return Type::getInt32Ty(TheContext);
  case  8: return Type::getInt64Ty(TheContext);
  case 16: return Type::getInt128Ty(TheContext);
  default:
    THROW_CONTRA_ERROR("Uknown id type with size " << sizeof(T));
  };
  return nullptr;
}

//==============================================================================
// Create the function wrapper
//==============================================================================
Function* LegionTasker::wrap(Module &TheModule, const std::string & Name,
    Function* TaskF) const
{
  auto VoidPtrType = llvmVoidPointerType(TheContext_);
  auto VoidType = Type::getVoidTy(TheContext_);
  auto SizeType = createIdType<std::size_t>(TheContext_);
  auto RealmIdType = createIdType<realm_id_t>(TheContext_);

  //----------------------------------------------------------------------------
  // Create task wrapper
  std::string TaskName = "__" + Name + "_task__";
  std::vector<Type *> WrapperArgTypes =
    {VoidPtrType, SizeType, VoidPtrType, SizeType, RealmIdType};
  
  FunctionType *WrapperT = FunctionType::get(VoidType, WrapperArgTypes, false);
  Function *WrapperF = Function::Create(WrapperT, Function::InternalLinkage,
      TaskName, &TheModule);
 
  auto Arg = WrapperF->arg_begin();
  Arg->setName("data");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("datalen");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("userdata");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("userlen");
  Arg->addAttr(Attribute::ReadOnly);
  Arg++;
  Arg->setName("procid");
  Arg->addAttr(Attribute::ReadOnly);

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext_, "entry", WrapperF);
  Builder_.SetInsertPoint(BB);

  // allocate arguments
  std::vector<Value*> WrapperArgsV;
  WrapperArgsV.reserve(WrapperArgTypes.size());

  IRBuilder<> TmpB(&WrapperF->getEntryBlock(), WrapperF->getEntryBlock().begin());

  unsigned ArgIdx = 0;
  for (auto &Arg : WrapperF->args()) {
    // get arg type
    auto ArgType = WrapperArgTypes[ArgIdx];
    // Create an alloca for this variable.
    auto Alloca = TmpB.CreateAlloca(ArgType, nullptr, Arg.getName());
    // Store the initial value into the alloca.
    Builder_.CreateStore(&Arg, Alloca);
    WrapperArgsV.emplace_back(Alloca);
    ArgIdx++;
  }

  // loads
  auto DataV = Builder_.CreateLoad(VoidPtrType, WrapperArgsV[0], "data");
  auto DataLenV = Builder_.CreateLoad(SizeType, WrapperArgsV[1], "datalen");
  //auto UserDataV = Builder_.CreateLoad(VoidPtrType, WrapperArgsV[2], "userdata");
  //auto UserLenV = Builder_.CreateLoad(SizeType, WrapperArgsV[3], "userlen");
  auto ProcIdV = Builder_.CreateLoad(RealmIdType, WrapperArgsV[4], "proc_id");

  //----------------------------------------------------------------------------
  // call to preamble

  // create temporaries
  auto OpaqueType = createOpaqueType("legion_opaque_t", TheContext_);
  auto OpaquePtrType = PointerType::get(OpaqueType, 0);
  
  auto TaskAlloca = TmpB.CreateAlloca(OpaqueType, nullptr, "task");
  
  auto RegionsAlloca = TmpB.CreateAlloca(OpaquePtrType, nullptr, "regions");
  auto NullV = Constant::getNullValue(OpaquePtrType);
  Builder_.CreateStore(NullV, RegionsAlloca);

  auto NumType = createIdType<std::uint32_t>(TheContext_); 
  auto NumRegionsAlloca = TmpB.CreateAlloca(NumType, nullptr, "num_regions");
  auto ZeroV = ConstantInt::get(TheContext_, APInt(32, 0 /* len for now */, false));  
  Builder_.CreateStore(ZeroV, NumRegionsAlloca);
  
  auto ContextAlloca = TmpB.CreateAlloca(OpaqueType, nullptr, "ctx");
  auto RuntimeAlloca = TmpB.CreateAlloca(OpaqueType, nullptr, "runtime");

  // args
  std::vector<Value*> PreambleArgsV = { DataV, DataLenV, ProcIdV,
    TaskAlloca, RegionsAlloca, NumRegionsAlloca, RuntimeAlloca, ContextAlloca };

  std::vector<Type*> PreambleArgTypes;
  PreambleArgTypes.reserve(PreambleArgsV.size());
  for (auto & Arg : PreambleArgsV) PreambleArgTypes.emplace_back( Arg->getType() );
  
  FunctionType *PreambleT = FunctionType::get(VoidType, PreambleArgTypes, false);
  Function *PreambleF = Function::Create(PreambleT, Function::ExternalLinkage,
      "legion_task_preamble", &TheModule);
  
  Builder_.CreateCall(PreambleF, PreambleArgsV, "preamble");

  //----------------------------------------------------------------------------
  // extrat user variables
  std::vector<Value*> TaskArgsV;
  
  //----------------------------------------------------------------------------
  // call users actual function
  Builder_.CreateCall(TaskF, TaskArgsV, "calltmp");

  //----------------------------------------------------------------------------
  // Postable
 
  // temporaries
  auto RuntimeV = Builder_.CreateLoad(OpaqueType, RuntimeAlloca, "runtime");
  auto ContextV = Builder_.CreateLoad(OpaqueType, ContextAlloca, "ctx");

  auto RetvalV = Constant::getNullValue(VoidPtrType);
  
  auto SizeBytes = sizeof(std::size_t) * 8;
  auto RetsizeV = ConstantInt::get(TheContext_, APInt(SizeBytes, 0, false));

  // args
  std::vector<Value*> PostambleArgsV = { RuntimeV, ContextV, RetvalV, RetsizeV };
  
  std::vector<Type*> PostambleArgTypes;
  PostambleArgTypes.reserve(PostambleArgsV.size());
  for (auto & Arg : PostambleArgsV) PostambleArgTypes.emplace_back( Arg->getType() );

  // call
  FunctionType *PostambleT = FunctionType::get(VoidType, PostambleArgTypes, false);
  Function *PostambleF = Function::Create(PostambleT, Function::ExternalLinkage,
      "legion_task_postamble", &TheModule);
  
  Builder_.CreateCall(PostambleF, PostambleArgsV, "preamble");
  
  //----------------------------------------------------------------------------
  // function retuns void
  Builder_.CreateRetVoid();

  return WrapperF;
}

}
