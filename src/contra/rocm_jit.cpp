#include "rocm_jit.hpp"
#include "rocm_rt.hpp"

#include "compiler.hpp"
#include "errors.hpp"
#include "utils/llvm_utils.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

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
  DeviceJIT(TheHelper)
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
}

//==============================================================================
// Create a new module
//==============================================================================
std::unique_ptr<Module> ROCmJIT::createModule()
{
  auto NewModule = std::make_unique<Module>("device jit", TheContext_);
  NewModule->setDataLayout(TargetMachine_->createDataLayout());
  NewModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
  return NewModule;
}

//==============================================================================
// Compile a module
//==============================================================================
void ROCmJIT::addModule(std::unique_ptr<Module> M) {

  //----------------------------------------------------------------------------
  // Add annotations
  
  //auto Annots = M->getOrInsertNamedMetadata("llvm.module.flags");
  //std::vector<Metadata*> Meta = {
  //  ValueAsMetadata::get( llvmValue<int>(TheContext_, 1) ),
  //  MDString::get(TheContext_, "wchar_size"),
  //  ValueAsMetadata::get( llvmValue<int>(TheContext_, 4) ) };
  //Annots->addOperand(MDNode::get(TheContext_, Meta));
  //Meta = {
  //  ValueAsMetadata::get( llvmValue<int>(TheContext_, 7) ),
  //  MDString::get(TheContext_, "PIC Level"),
  //  ValueAsMetadata::get( llvmValue<int>(TheContext_, 1) ) };
  //Annots->addOperand(MDNode::get(TheContext_, Meta));

  //Annots = M->getOrInsertNamedMetadata("opencl.ocl.version");
  //Meta = {
  //  ValueAsMetadata::get( llvmValue<int>(TheContext_, 2) ),
  //  ValueAsMetadata::get( llvmValue<int>(TheContext_, 0) ) };
  //Annots->addOperand(MDNode::get(TheContext_, Meta));
  
  //----------------------------------------------------------------------------
  // Fix globals
  for (auto & GV : M->getGlobalList()) {
    auto Ty = GV.getType()->getPointerElementType();
    //GV.mutateType(PointerType::get(Ty, 4));
    //GV.setLinkage(GlobalValue::InternalLinkage);
    //GV.setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    //auto NewGV = new GlobalVariable(
    //  *M,
    //  Ty,
    //  true,
    //  GlobalValue::PrivateLinkage,
    //  GV.getInitializer(),
    //  "",
    //  &GV,
    //  GlobalValue::NotThreadLocal,
    //  4);
    //GV.getType()->print(outs()); outs()<<" ";
    //NewGV->getType()->print(outs()); outs()<<"\n";
    //GV.replaceAllUsesWith(NewGV); 
  }

  
  //----------------------------------------------------------------------------
  // Replace calls/intrinsics

  std::vector<std::string> KernelNames;

  for (auto & Func : M->getFunctionList()) {
    
    if (Func.getInstructionCount())
      KernelNames.emplace_back(Func.getName());

    for (auto BlockIt=Func.begin(); BlockIt!=Func.end(); ++BlockIt) {
      for (auto InstIt=BlockIt->begin(); InstIt!=BlockIt->end(); ++InstIt) {
        Instruction* NewI = nullptr;

        //------------------------------
        // Replace Calls
        if (auto CallI = dyn_cast<CallInst>(InstIt)) {
          auto CallF = CallI->getCalledFunction();

          if (CallF->getName().str() == "print") {
            //auto BB = replacePrint(*M, CallI);
            //BlockIt = Function::iterator(BB);
            //break;
            NewI = replacePrint2(*M, CallI);
          }

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

  //----------------------------------------------------------------------------
  // Compile
  
  std::string temp_name = std::tmpnam(nullptr);
 
  runOnModule(*M);
  
  auto NewM = insertBitcode(*M, temp_name);
  
  auto Hsaco = compileAndLink(*NewM, temp_name);

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
      NumKernels);

  delete[] KernelNamesC;

}

//==============================================================================
// Compile a module by cloning it first
//==============================================================================
void ROCmJIT::addModule(const Module * M) {
  auto ClonedModule = CloneModule(*M);

  ClonedModule->setSourceFileName("device jit");
  ClonedModule->setDataLayout(TargetMachine_->createDataLayout());
  ClonedModule->setTargetTriple(TargetMachine_->getTargetTriple().getTriple());
  addModule(std::move(ClonedModule));
}

//==============================================================================
// Verify module
//==============================================================================
void ROCmJIT::runOnModule(Module & TheModule)
{
  auto PassMan = legacy::PassManager();
  PassMan.add(createVerifierPass());
  
  auto LLVMT = static_cast<LLVMTargetMachine*>(TargetMachine_);
  TargetPassConfig * Config = LLVMT->createPassConfig(PassMan);
  Config->addIRPasses();
  PassMan.add(Config);
  
  PassMan.run(TheModule);
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
    Dest = std::make_unique<raw_fd_ostream>(Filename, EC, sys::fs::OF_None);
    if (EC)
      THROW_CONTRA_ERROR( "Could not open file: " << EC.message() );
  }
  
  //----------------------------------------------------------------------------
  // Compile
  
  auto PassMan = legacy::PassManager();
  PassMan.add(createVerifierPass());

  auto LLVMT = static_cast<LLVMTargetMachine*>(TargetMachine_);
  TargetPassConfig * Config = LLVMT->createPassConfig(PassMan);
  Config->addIRPasses();
  PassMan.add(Config);

  auto fail = TargetMachine_->addPassesToEmitFile(
      PassMan,
      *Dest,
      nullptr,
      FileType,
      false);
  if (fail)
    THROW_CONTRA_ERROR( "Error generating PTX");
  
  PassMan.run(TheModule);

  return SmallStr.str();

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
std::unique_ptr<Module> ROCmJIT::insertBitcode(
    Module & M,
    std::string file_name)
{
  auto orig_name = file_name + ".bc";
  auto new_name = file_name + ".linked.bc";

  std::error_code EC;
  raw_fd_ostream File(orig_name, EC, sys::fs::OF_None);
  WriteBitcodeToFile(M, File);
  File.flush();
  File.close();

  std::stringstream ss;
  ss << ROCM_LLVM_LINK_PATH << " "
    << orig_name 
    << " /opt/rocm-3.5.0/lib/ockl.amdgcn.bc "
       "/opt/rocm-3.5.0/lib/opencl.amdgcn.bc -o "
    << new_name;

  auto res = system( ss.str().c_str() );
  if (res) THROW_CONTRA_ERROR("Error linking bitcode.");

  SMDiagnostic Diag;
  auto NewM = parseIRFile(
      new_name,
      Diag,
      TheContext_);
  if (!NewM) 
    THROW_CONTRA_ERROR("Error reading new bitcode file.");

  return NewM;
}

//==============================================================================
// Helper to replace print function
//==============================================================================
BasicBlock* ROCmJIT::replacePrint(Module &M, CallInst* CallI) {

  // some types
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto Int32T = Type::getInt32Ty(TheContext_);

  // create new print function
  auto PrintT = FunctionType::get(
      Int32T,
      VoidPtrT,
      true /* var args */ );
  auto PrintF = M.getOrInsertFunction("printf", PrintT).getCallee();

  // gather args
  std::vector<Value*> ArgVs;
  for (auto & A : CallI->args())
    ArgVs.push_back(A);


  // create new instruction            
  auto TmpB = IRBuilder<>(CallI);
  emitAMDGPUPrintfCall(TmpB, ArgVs);

  // erase old
  CallI->eraseFromParent();

  return TmpB.GetInsertBlock();
  
}

//==============================================================================
// Helper to replace print function
//==============================================================================
CallInst* ROCmJIT::replacePrint2(Module &M, CallInst* CallI) {

  // some types
  auto VoidPtrT = llvmType<void*>(TheContext_);
  auto Int32T = Type::getInt32Ty(TheContext_);

  // create new print function
  auto PrintT = FunctionType::get(
      Int32T,
      VoidPtrT,
      true /* var args */ );
  auto PrintF = M.getOrInsertFunction("printf", PrintT).getCallee();
    
  // gather args
  std::vector<Value*> ArgVs;
  for (auto & Arg : CallI->args()) ArgVs.push_back(Arg);

  // create new instruction            
  auto TmpB = IRBuilder<>(TheContext_);
  return TmpB.CreateCall(PrintF, ArgVs, CallI->getName());
}


} // namepsace
