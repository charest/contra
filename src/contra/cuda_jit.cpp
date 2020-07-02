#include "cuda_jit.hpp"

#include "cuda_rt.hpp"
#include "compiler.hpp"
#include "errors.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <set>

using namespace llvm;
using namespace utils;

namespace contra {

//==============================================================================
// The constructor
//==============================================================================
CudaJIT::CudaJIT(BuilderHelper & TheHelper) :
  TheHelper_(TheHelper),
  Builder_(TheHelper.getBuilder()),
  TheContext_(TheHelper.getContext())
{
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  auto Tgt = utils::findTarget(TargetNvptxString);
  if (!Tgt) 
    THROW_CONTRA_ERROR(
        "Cuda backend selected but LLVM does not support 'nvptx64'");
  Triple Trip;
  Trip.setArch(Triple::nvptx64);
  Trip.setVendor(Triple::NVIDIA);
  Trip.setOS(Triple::CUDA);
  TargetMachine_ = Tgt->createTargetMachine(
        Trip.getTriple(),
        "sm_20",
        "",
        TargetOptions(),
        None,
        None,
        CodeGenOpt::Aggressive);
  contra_cuda_startup();
}

//==============================================================================
// The destructor
//==============================================================================
CudaJIT::~CudaJIT()
{
  contra_cuda_shutdown();
}
 
//==============================================================================
// Create a new module
//==============================================================================
std::unique_ptr<Module> CudaJIT::createModule()
{
  auto NewModule = std::make_unique<Module>("devicee jit", TheContext_);
  NewModule->setDataLayout(TargetMachine_->createDataLayout());
  NewModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
  return NewModule;
}

//==============================================================================
// Compile a module
//==============================================================================
void CudaJIT::addModule(std::unique_ptr<Module> M) {
  auto DL = TargetMachine_->createDataLayout();

  for (auto & Func : M->getFunctionList()) {
    for (auto & Block : Func.getBasicBlockList()) {
      for (auto InstIt=Block.begin(); InstIt!=Block.end(); ++InstIt) {
        Instruction* NewI = nullptr;

        //----------------------------------------------------------------------
        // Replace Calls
        if (auto CallI = dyn_cast<CallInst>(InstIt)) {
          auto CallF = CallI->getCalledFunction();

          if (CallF->getName().str() == "print")
            NewI = replacePrint(*M, CallI);

        } // call
        // Done Replacements
        //----------------------------------------------------------------------
            
        // replace instruction
        if (NewI) {
          ReplaceInstWithInst(&(*InstIt), NewI);
          InstIt = BasicBlock::iterator(NewI);
        }
      } // Instruction
    } // Block
  
    verifyFunction(Func);

  } // Function

  //M->print(outs(), nullptr); outs() << "\n";
  auto KernelStr = compileKernel(
      *M,
      TargetMachine_);
  contra_cuda_register_kernel(KernelStr.c_str());
}

//==============================================================================
// Compile a module by cloning it first
//==============================================================================
void CudaJIT::addModule(const Module * M) {
    std::set<GlobalValue *> ClonedDefsInSrc;
    ValueToValueMapTy VMap;
    auto ClonedModule = CloneModule(
        *M,
        VMap,
        [](const GlobalValue *GV) { return true; });

    ClonedModule->setSourceFileName("device jit");
    ClonedModule->setDataLayout(TargetMachine_->createDataLayout());
    ClonedModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
    addModule(std::move(ClonedModule));
}

//==============================================================================
// Helper to replace print function
//==============================================================================
CallInst* CudaJIT::replacePrint(Module &M, CallInst* CallI) {

  // some types
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto Int32T = Type::getInt32Ty(TheContext_);

  // create new print function
  auto PrintT = FunctionType::get(
      Int32T,
      {VoidPtrT, VoidPtrT},
      false /* var args */ );
  auto PrintF = M.getOrInsertFunction("vprintf", PrintT).getCallee();
  auto NumArgs = CallI->arg_size();
  Value* VarArgsV;

  // No Args!
  if (NumArgs <= 1) {
    VarArgsV = Constant::getNullValue(VoidPtrT);

  }
  // Has Args
  else {

    // gather args
    std::vector<Value*> ArgVs;
    std::vector<Type*> ArgTs;
    auto ArgIt = CallI->arg_begin();
    ArgIt++;
    for (; ArgIt != CallI->arg_end(); ++ArgIt) {
      ArgVs.push_back(ArgIt->get());
      ArgTs.push_back(ArgIt->get()->getType());
    }

    // create a struct
    auto ParentF = CallI->getParent()->getParent();
    auto AllocaT = StructType::create(ArgTs, "printf_args");
    auto AllocaA = TheHelper_.createEntryBlockAlloca(ParentF, AllocaT);

    // store values as memebers
    Value *Idxs[] = {
      ConstantInt::get(Int32T, 0),
      ConstantInt::get(Int32T, 0)};

    for (unsigned i=0; i<NumArgs-1; ++i) {    
      Idxs[1] = ConstantInt::get(Int32T, i);
      auto GEP = GetElementPtrInst::CreateInBounds(
        AllocaT,
        AllocaA,
        Idxs,
        "printf_args.i",
        CallI);
      new StoreInst(ArgVs[i], GEP, CallI);
    }

    // cast struct type
    VarArgsV = CastInst::Create(
      Instruction::BitCast,
      AllocaA,
      VoidPtrT,
      "cast",
      CallI);
  }

  // create new instruction            
  auto ArgIt = CallI->arg_begin();
  std::vector <Value*> ArgVs = {ArgIt->get(), VarArgsV};
  auto TmpB = IRBuilder<>(TheContext_);
  return TmpB.CreateCall(PrintF, ArgVs, CallI->getName());
}

} // namepsace
