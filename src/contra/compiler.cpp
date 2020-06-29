#include "compiler.hpp"
#include "errors.hpp"

#include "utils/llvm_utils.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Standard compiler for host
////////////////////////////////////////////////////////////////////////////////
void compile(Module & TheModule, const std::string & Filename) {
  
  utils::initializeAllTargets();

  auto TargetTriple = sys::getDefaultTargetTriple();
  TheModule.setTargetTriple(TargetTriple);

  std::string Error;
  auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);

  // Print an error and exit if we couldn't find the requested target.
  // This generally occurs if we've forgotten to initialise the
  // TargetRegistry or we have a bogus target triple.
  if (!Target) {
    THROW_CONTRA_ERROR( Error );
  }

  auto CPU = "generic";
  auto Features = "";

  TargetOptions opt;
  auto RM = Optional<Reloc::Model>();
  auto TheTargetMachine =
      Target->createTargetMachine(TargetTriple, CPU, Features, opt, RM);

  TheModule.setDataLayout(TheTargetMachine->createDataLayout());

  std::error_code EC;
  raw_fd_ostream dest(Filename, EC, sys::fs::OF_None);

  if (EC)
    THROW_CONTRA_ERROR( "Could not open file: " << EC.message() );

  legacy::PassManager pass;
  auto FileType = CGFT_ObjectFile;

  if (TheTargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
    THROW_CONTRA_ERROR( "TheTargetMachine can't emit a file of this type" );
  }

  pass.run(TheModule);
  dest.flush();

  std::cout << "Wrote " << Filename << "\n";

}

////////////////////////////////////////////////////////////////////////////////
// Standard compiler for host
////////////////////////////////////////////////////////////////////////////////
std::string compileKernel(
    Module & TheModule,
    TargetMachine * TM,
    const std::string & Filename) 
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

  auto fail = TM->addPassesToEmitFile(
      PassMan,
      *Dest,
      nullptr,
      CGFT_AssemblyFile,
      false);
  if (fail)
    THROW_CONTRA_ERROR( "Error generating PTX");
  
  PassMan.run(TheModule);

  return SmallStr.str();

}

} // namespace
