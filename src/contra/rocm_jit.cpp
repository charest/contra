#include "rocm_jit.hpp"
#include "rocm_rt.hpp"

#include "compiler.hpp"
#include "errors.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

#include "llvm-ext/Target/AMDGPU/AMDGPU.h"

#include <fstream>
#include <set>

using namespace llvm;
using namespace utils;

namespace contra {

//==============================================================================
// The constructor
//==============================================================================
ROCmJIT::ROCmJIT(BuilderHelper & TheHelper) :
  DeviceJIT(TheHelper),
  LinkFlags_(Linker::Flags::OverrideFromSrc)
{
  std::string CPU = hasTargetCPU() ?
    getTargetCPU() : std::string(ROCM_DEFAULT_TARGET_CPU);

  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  auto Tgt = utils::findTarget("amdgcn");
  if (!Tgt) 
    THROW_CONTRA_ERROR(
        "ROCm backend selected but LLVM does not support 'amdgcn'");
  Triple Trip;
  Trip.setArch(Triple::amdgcn);
  Trip.setVendor(Triple::AMD);
  Trip.setOS(Triple::AMDHSA);
  TargetMachine_ = Tgt->createTargetMachine(
        Trip.getTriple(),
        CPU,
        "", //"-code-object-v3",
        TargetOptions(),
        std::nullopt,
        std::nullopt,
        CodeGenOptLevel::Aggressive);
  
  contra_rocm_startup();
  
  if (hasMaxBlockSize())
    contra_rocm_set_block_size(getMaxBlockSize());
}

//==============================================================================
// Create a new module
//==============================================================================
std::unique_ptr<Module> ROCmJIT::createModule(const std::string & Name)
{
  std::string NewName = Name.empty() ? "Device jit" : Name;
  auto NewModule = std::make_unique<Module>(NewName, TheContext_);
  NewModule->setDataLayout(TargetMachine_->createDataLayout());
  NewModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
  return NewModule;
}
  
//==============================================================================
// Compile a module by cloning it first
//==============================================================================
std::unique_ptr<Module> ROCmJIT::cloneModule(const Module & M) {
  auto ClonedModule = CloneModule(M);

  ClonedModule->setSourceFileName(M.getName());
  ClonedModule->setDataLayout(TargetMachine_->createDataLayout());
  ClonedModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());

  return ClonedModule;
}

//==============================================================================
// Compile a module by cloning it first
//==============================================================================
void ROCmJIT::addModule(const Module * M) {
  auto ClonedModule = cloneModule(*M);

  runOnModule(*ClonedModule);

  // must be just a function since we are not given
  // ownership of module
  UserModules_.emplace_back( std::move(ClonedModule) );
}


//==============================================================================
// Compile a module
//==============================================================================
void ROCmJIT::addModule(std::unique_ptr<Module> M) {
  
  //----------------------------------------------------------------------------
  // Sanitize

  runOnModule(*M);
  
  std::vector<std::string> KernelNames;
  for (auto & Func : M->getFunctionList()) {
    if (Func.getInstructionCount() &&
        Func.getCallingConv() == CallingConv::AMDGPU_KERNEL )
      KernelNames.emplace_back(Func.getName());
  }

  //----------------------------------------------------------------------------
  // Compile
  
  bool HasReduce = M->getFunction("apply");

  assemble(*M, HasReduce);
  auto Hsaco = compileAndLink(*M);

  //----------------------------------------------------------------------------
  // Register

  auto NumKernels = KernelNames.size();
  auto KernelNamesC = new const char*[NumKernels];
  for (unsigned i=0; i<NumKernels; ++i)
    KernelNamesC[i] = KernelNames[i].c_str();

  contra_rocm_register_kernel(
      Hsaco.data(),
      Hsaco.size(),
      KernelNamesC,
      NumKernels,
      HasReduce);

  delete[] KernelNamesC;

}


//==============================================================================
// Verify module
//==============================================================================
void ROCmJIT::runOnModule(Module & M)
{
  //----------------------------------------------------------------------------
  // Replace calls/intrinsics
here:
  for (auto & Func : M.getFunctionList()) {
    for (auto BlockIt=Func.begin(); BlockIt!=Func.end(); ++BlockIt) {
      for (auto InstIt=BlockIt->begin(); InstIt!=BlockIt->end(); ++InstIt) {
        Instruction* NewI = nullptr;

        //------------------------------
        // Replace Calls
        if (auto CallI = dyn_cast<CallInst>(InstIt)) {
          auto CallF = CallI->getCalledFunction();
          auto CallN = CallF->getName().str();
          auto RetT = CallF->getReturnType();
						
#if 1
          FunctionType *FTy = CallF->getFunctionType();
          for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i) {
            if (CallI->getArgOperand(i)->getType() != FTy->getParamType(i)) {
          	  std::cout << "Function '" << CallN << "' args[" << i << "] dont match" << std::endl;
              CallI->print(outs()); outs() << "\n";
              FTy->print(outs()); outs() << "\n";
            }
          }
#endif

          if (CallN == "print") {
            NewI = replaceName(M, CallI, "printf");
            //auto Res = replacePrint2(M, CallI);
						//NewI = dyn_cast<Instruction>(Res);
						//std::cout << "NewI " << NewI << std::endl;	
          }
          else if (CallN == "sqrt") {
            NewI = replaceIntrinsic(M, CallI, Intrinsic::sqrt, {RetT}); 
          }
          else if (CallF->getName().str() == "fabs") {
            NewI = replaceIntrinsic(M, CallI, Intrinsic::fabs, {RetT});
          }
          else if (CallN == "fmax")
            NewI = replaceIntrinsic(M, CallI, Intrinsic::maxnum, {RetT});
          else if (CallN == "fmin")
            NewI = replaceIntrinsic(M, CallI, Intrinsic::minnum, {RetT});
          else if (CallN == "__syncthreads")
            NewI = replaceSync(M, CallI);

        } // call
        // Done Replacements
        //------------------------------
            
        // replace instruction
        if (NewI) {
          if (InstIt->getParent())
            ReplaceInstWithInst(&(*InstIt), NewI);
          InstIt = BasicBlock::iterator(NewI);
					//goto here;
        }
      } // Instruction
    } // Block
  
    verifyFunction(Func);

  } // Function
  
}

//==============================================================================
// Standard compiler for host
//==============================================================================
void ROCmJIT::compile(
    Module & TheModule,
    raw_pwrite_stream & Dest)
{

  auto PassMan = legacy::PassManager();

  PassMan.add(createVerifierPass());

  auto fail = TargetMachine_->addPassesToEmitFile(
      PassMan,
      Dest,
      nullptr,
      CodeGenFileType::ObjectFile,
      false);
  if (fail)
    THROW_CONTRA_ERROR( "Error generating PTX");
  
  PassMan.run(TheModule);

}

//==============================================================================
// Compile a module
//==============================================================================
std::vector<char> ROCmJIT::compileAndLink(Module & M)
{

  // compile to isa
  auto IsaFile =  sys::fs::TempFile::create("contra-%%%%%%%.isabin");
  if (!IsaFile)
    THROW_CONTRA_ERROR( "Could not create temporary file." );
  auto IsaName = IsaFile->TmpName;
  
  raw_fd_ostream IsaFS(IsaFile->FD, false);
  compile(M, IsaFS);
  IsaFS.flush();
  
  // convert to hsaco
  SmallString<128> HsacoName;
  sys::fs::createUniquePath(
      "contra-%%%%%%%.hsaco",
      HsacoName,
      true);

  std::vector<StringRef> LdArgs {
    ROCM_LD_LLD_PATH,
    "-flavor",
    "gnu",
    "-shared",
    IsaName,
    "-o",
    HsacoName
  };

  std::string error_message;
  auto res = sys::ExecuteAndWait(
    ROCM_LD_LLD_PATH,
    LdArgs,
    std::nullopt,
    {},
    0,
    0,
    &error_message);
  
  if (res) {
    THROW_CONTRA_ERROR(
        "ld.lld execute failed: '" << error_message << "', error code: "
        << res);
  }
  
  // discard file
  if (auto Err = IsaFile->discard())
    THROW_CONTRA_ERROR( "Could not discard temporary file." );


  // read hsaco
  std::ifstream file(HsacoName.c_str(), std::ios::binary | std::ios::ate);
  auto size = file.tellg();

  std::vector<char> HsacoBytes(size);
  file.seekg(0, std::ios::beg);
  file.read(HsacoBytes.data(), size);
  file.close();
  
  return HsacoBytes;

}

//==============================================================================
// Compile a module
//==============================================================================
void ROCmJIT::assemble(Module & M, bool HasReduce)
{

  Linker L(M);

  //----------------------------------------------------------------------------
  // Add user device modules

  for (auto & UserM : UserModules_) {
    auto ClonedM = cloneModule(*UserM);
    auto err = L.linkInModule(std::move(ClonedM), LinkFlags_);
    if (err) THROW_CONTRA_ERROR("Error linking in user device module.");
  }
  
  if (HasReduce) {
    auto CPUStr = TargetMachine_->getTargetCPU().str();
    linkFiles(
        L, 
        {
          ROCM_DEVICE_USER_PATH"/librtrocm/rocm_scratch." + CPUStr + ".bc",
          ROCM_DEVICE_USER_PATH"/contra/rocm_reduce." + CPUStr + ".bc"
        }, 
        LinkFlags_);
    runOnModule(M);
  }
      
  //----------------------------------------------------------------------------
  // Fix functoin attributes

  for (auto & Func : M) {
    if (!Func.isDeclaration()) {
      if (Func.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
        Func.removeFnAttr(llvm::Attribute::OptimizeNone);
      }
      else {
        Func.setLinkage(GlobalValue::LinkOnceODRLinkage);
        Func.setVisibility(GlobalValue::ProtectedVisibility);
        Func.removeFnAttr(Attribute::OptimizeNone);
        Func.removeFnAttr(Attribute::NoInline);
        Func.addFnAttr(Attribute::AlwaysInline);
      }
    }
  }

  //----------------------------------------------------------------------------
  // Run passmanager
  // Replace printfs, and fix addresses

  auto PassMan = legacy::PassManager();
  
  //PassMan.add(createAMDGPUPrintfRuntimeBinding());
  PassMan.add(createInferAddressSpacesPass());
  PassMan.add(createSROAPass());
  //PassMan.add(createAMDGPULowerAllocaPass());
  
  PassMan.run(M);

  bool NeedsPrint = M.getNamedMetadata("llvm.printf.fmts");

  //----------------------------------------------------------------------------
  // Add bytecode
      
  std::vector<std::string> Files = {
      OCLC_DAZ_OPT_OFF_PATH,
      OCLC_UNSAFE_MATH_OFF_PATH,
      OCLC_FINITE_ONLY_OFF_PATH,
      OCLC_CORRECTLY_ROUNDED_SQRT_ON_PATH,
      OCLC_WAVEFRONTSIZE64_ON_PATH,
      OCLC_ISA_VERSION_942_PATH,
    };
  linkFiles(L, Files, LinkFlags_);
  
  if (NeedsPrint) {
    linkFiles(
        L, 
        {ROCM_DEVICE_LIB_PATH"/opencl.amdgcn.bc"}, 
        LinkFlags_ & Linker::Flags::LinkOnlyNeeded);
  }

  // verify
  auto error_message = verifyModule(M);
  if (!error_message.empty())
    THROW_CONTRA_ERROR("Linked module is broken: " << error_message);

}

//==============================================================================
// Link in bitcode
//==============================================================================
void ROCmJIT::linkFiles(
    Linker & L,
    const std::vector<std::string>& Files,
    unsigned Flags)
{
  for (const auto & F : Files) {
    SMDiagnostic Diag;
    auto DeviceM = getLazyIRFileModule(
        F,
        Diag,
        TheContext_);
    if (!DeviceM) {
      std::cerr << Diag.getMessage().str() << std::endl;
      THROW_CONTRA_ERROR("Error reading device bitcode file: " << F);
    }

    auto err = L.linkInModule(std::move(DeviceM), Flags);
    if (err)
      THROW_CONTRA_ERROR("Error linking in device module.");
  }
}

//==============================================================================
// Link in bitcode
//==============================================================================
Instruction* ROCmJIT::replaceSync(Module & M, CallInst * CallI){
  auto WorkgroupSSID = TheContext_.getOrInsertSyncScopeID("workgroup");
  IRBuilder<> TmpB(CallI);
  TmpB.CreateFence(AtomicOrdering::Release, WorkgroupSSID);
  auto F = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_s_barrier); 
  TmpB.CreateCall(F);
  auto FenceI = TmpB.CreateFence(AtomicOrdering::Acquire, WorkgroupSSID);
  CallI->eraseFromParent();
  return FenceI;
}

//==============================================================================
// Helper to replace print function
//==============================================================================
CallInst* ROCmJIT::replacePrint(Module &M, CallInst* CallI) {
  
  // gather args
  std::vector<Value*> ArgVs;
  for (auto & Arg : CallI->args()) ArgVs.push_back(Arg.get());
        
	auto CallF = CallI->getCalledFunction();
	auto FTy = CallF->getFunctionType();
								
	outs() << "\nHERHEEHER "; CallI->print(outs()); outs() << "\n";
	outs() << "\nHERHEEHER "; FTy->print(outs()); outs() << "\n";
	
	outs() << "\nHERHEEHER "; ArgVs[0]->getType()->print(outs()); outs() << "\n";
	outs() << "\nHERHEEHER "; FTy->getParamType(0)->print(outs()); outs() << "\n";

	if (ArgVs[0]->getType() != FTy->getParamType(0)) {
  	// cast struct type
  	ArgVs[0] = CastInst::Create(
  	  Instruction::AddrSpaceCast,
  	  ArgVs[0],
  	  FTy->getParamType(0),
  	  "cast",
  	  CallI);
		outs() << "\nHERHEEHER "; ArgVs[0]->getType()->print(outs()); outs() << "\n";
		outs() << "\nHERHEEHER "; ArgVs[0]->print(outs()); outs() << "\n";

	}

  // create new instruction            
  auto TmpB = IRBuilder<>(TheContext_);

  Function * NewF = M.getFunction("printf");
  if (!NewF) {  
    NewF = Function::Create(
        FTy,
        CallF->getLinkage(),
        CallF->getAddressSpace(),
        "printf",
        &M);
  }
  return TmpB.CreateCall(NewF, ArgVs, CallI->getName());
}

Value* ROCmJIT::replacePrint2(Module &M, CallInst* CallI) {

  // gather args
  std::vector<Value*> ArgVs;
  for (auto & A : CallI->args())
    ArgVs.push_back(A);

	auto CallF = CallI->getCalledFunction();
	auto FTy = CallF->getFunctionType();
	if (ArgVs[0]->getType() != FTy->getParamType(0)) {
  	// cast struct type
  	ArgVs[0] = CastInst::Create(
  	  Instruction::AddrSpaceCast,
  	  ArgVs[0],
  	  FTy->getParamType(0),
  	  "cast",
  	  CallI);
		outs() << "\nHERHEEHER "; ArgVs[0]->getType()->print(outs()); outs() << "\n";
		outs() << "\nHERHEEHER "; ArgVs[0]->print(outs()); outs() << "\n";

	}

  // create new instruction
  IRBuilder<> TmpB(CallI);
  auto ResV = emitAMDGPUPrintfCall(TmpB, ArgVs, false);

  // erase old
  //CallI->eraseFromParent();

	return ResV;

}

} // namepsace
