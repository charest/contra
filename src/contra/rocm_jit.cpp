#include "rocm_jit.hpp"
#include "rocm_rt.hpp"

#include "compiler.hpp"
#include "errors.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm-ext/Target/AMDGPU/AMDGPU.h"                                      
#include "llvm-ext/Transforms/Utils/AMDGPUEmitPrintf.h"                                

#include <fstream>
#include <set>
#include <unistd.h>

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
        "gfx900",
        "", //"-code-object-v3",
        TargetOptions(),
        None,
        None,
        CodeGenOpt::Aggressive);

  UserModule_ = createModule("function lib");
  
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
  Linker::linkModules(*UserModule_, std::move(ClonedModule), LinkFlags_);
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
  
  std::string temp_name = std::tmpnam(nullptr);
  assemble(*M, temp_name, HasReduce);
  auto Hsaco = compileAndLink(*M, temp_name);


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

          if (CallN == "print") {
            NewI = replaceName(M, CallI, "printf");
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

        } // call
        // Done Replacements
        //------------------------------
            
        // replace instruction
        if (NewI) {
          ReplaceInstWithInst(&(*InstIt), NewI);
          InstIt = BasicBlock::iterator(NewI);
        }
      } // Instruction
    } // Block
  
    verifyFunction(Func);

  } // Function
  
}

//==============================================================================
// Standard compiler for host
//==============================================================================
std::string ROCmJIT::compile(
    Module & TheModule,
    const std::string & Filename,
    CodeGenFileType FileType)
{

  if (!TheModule.getInstructionCount()) return "";
  
  //----------------------------------------------------------------------------
  // Create output stream

  std::unique_ptr<raw_pwrite_stream> Dest;
  
  SmallString<SmallVectorLength> SmallStr;

  // output to string
  if (Filename.empty()) {
    Dest = std::make_unique<raw_svector_ostream>(SmallStr);
  }
  // output to file
  else {
    std::error_code EC;
    Dest = std::make_unique<raw_fd_ostream>(Filename, EC);
    if (EC)
      THROW_CONTRA_ERROR( "Could not open file: " << EC.message() );
  }
  
  //----------------------------------------------------------------------------
  // Compile
  
  auto PassMan = legacy::PassManager();

  PassMan.add(createVerifierPass());

  auto fail = TargetMachine_->addPassesToEmitFile(
      PassMan,
      *Dest,
      nullptr,
      FileType,
      false);
  if (fail)
    THROW_CONTRA_ERROR( "Error generating PTX");
  
  PassMan.run(TheModule);

  return SmallStr.str().str();

}

//==============================================================================
// Compile a module
//==============================================================================
std::vector<char> ROCmJIT::compileAndLink(
    Module & M,
    const std::string & file_name)
{

  // compile to isa
  auto isa_name = file_name + ".isabin";
  compile(M, isa_name, CGFT_ObjectFile);

  // convert to hsaco
  auto hsaco_name = file_name + ".hsaco";
  std::vector<StringRef> LdArgs {
    ROCM_LD_LLD_PATH,
    "-flavor",
    "gnu",
    "-shared",
    isa_name,
    "-o",
    hsaco_name
  };

  std::string error_message;
  auto res = sys::ExecuteAndWait(
    ROCM_LD_LLD_PATH,
    LdArgs,
    None,
    {},
    0,
    0,
    &error_message);
  
  if (res) {
    THROW_CONTRA_ERROR(
        "ld.lld execute failed: '" << error_message << "', error code: "
        << res);
  }

  // read hsaco
  std::ifstream file(hsaco_name, std::ios::binary | std::ios::ate);
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
void ROCmJIT::assemble(
    Module & M,
    std::string file_name,
    bool HasReduce)
{

  Linker L(M);

  //----------------------------------------------------------------------------
  // Add user device modules

  auto ClonedM = cloneModule(*UserModule_);
  auto err = L.linkInModule(std::move(ClonedM), Linker::Flags::LinkOnlyNeeded);
  if (err) THROW_CONTRA_ERROR("Error linking in user device module.");
  
  if (HasReduce) {
    linkFiles(
        L, 
        {
          ROCM_DEVICE_USER_PATH"/librtrocm/rocm_scratch.bc",
          ROCM_DEVICE_USER_PATH"/contra/rocm_reduce.bc"
        }, 
        LinkFlags_);
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
  
  PassMan.add(createAMDGPUPrintfRuntimeBinding());
  PassMan.add(createInferAddressSpacesPass());
  PassMan.add(createSROAPass());
  PassMan.add(createAMDGPULowerAllocaPass());
  
  PassMan.run(M);

  bool NeedsPrint = M.getNamedMetadata("llvm.printf.fmts");

  //----------------------------------------------------------------------------
  // Add bytecode
      
  std::vector<std::string> Files = {
      ROCM_DEVICE_LIB_PATH"/oclc_daz_opt_off.amdgcn.bc",
      ROCM_DEVICE_LIB_PATH"/oclc_unsafe_math_off.amdgcn.bc",
      ROCM_DEVICE_LIB_PATH"/oclc_finite_only_off.amdgcn.bc",
      ROCM_DEVICE_LIB_PATH"/oclc_correctly_rounded_sqrt_on.amdgcn.bc",
      ROCM_DEVICE_LIB_PATH"/oclc_wavefrontsize64_on.amdgcn.bc",
      ROCM_DEVICE_LIB_PATH"/oclc_isa_version_900.amdgcn.bc",
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

} // namepsace
