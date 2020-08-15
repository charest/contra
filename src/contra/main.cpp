#include "config.hpp"

#include "args.hpp"
#include "contra.hpp"
#include "precedence.hpp"

#include "librt/librt.hpp"

#include "utils/string_utils.hpp"

#include <mpi.h>

#include <fstream>
#include <iostream>
#include <memory>

using namespace contra;
using namespace utils;
using namespace llvm;

//==============================================================================
// Options
//==============================================================================

llvm::cl::opt<bool> OptionVerbose(
    "verbose",
    llvm::cl::desc("Print extra information"),
    llvm::cl::cat(OptionCategory));
llvm::cl::alias OptionVerboseA(
    "v",
    llvm::cl::desc("Alias for -verbose"),
    llvm::cl::aliasopt(OptionVerbose));

llvm::cl::opt<bool> OptionForce(
    "force",
    llvm::cl::desc("Force overwriting of existing files"),
    llvm::cl::cat(OptionCategory));
llvm::cl::alias OptionForceA(
    "f",
    llvm::cl::desc("Alias for -force"),
    llvm::cl::aliasopt(OptionForce));

llvm::cl::opt<bool> OptionCompile(
    "c",
    llvm::cl::desc("Compile the source files"),
    llvm::cl::cat(OptionCategory));

llvm::cl::opt<std::string> OptionOutput(
    "o",
    llvm::cl::desc("Place compiled output in <filename>"),
    llvm::cl::value_desc("filename"),
    llvm::cl::cat(OptionCategory));

llvm::cl::opt<bool> OptionDebug(
    "g",
    llvm::cl::desc("Produce debugging information"),
    llvm::cl::cat(OptionCategory));

enum OptLevel {
  O0, O1, O2, O3
};

llvm::cl::opt<OptLevel> OptionOptimizationLevel(
  llvm::cl::desc("Choose optimization level:"),
  llvm::cl::values(
    clEnumVal(O0 , "No optimizations"),
    clEnumVal(O1, "Enable trivial optimizations"),
    clEnumVal(O2, "Enable default optimizations"),
    clEnumVal(O3, "Enable expensive optimizations")),
  llvm::cl::init(O0),
  llvm::cl::cat(OptionCategory));

llvm::cl::opt<std::string> OptionDumpIR(
  "dump-ir",
  llvm::cl::desc("Dump LLVM IR to <filename>"),
  llvm::cl::value_desc("filename"),
  llvm::cl::cat(OptionCategory));

llvm::cl::opt<std::string> OptionDumpDot(
  "dump-dot",
  llvm::cl::desc("Dump AST in graphviz format to <filename>"),
  llvm::cl::value_desc("filename"),
  llvm::cl::cat(OptionCategory));

llvm::cl::opt<std::string> OptionBackend(
  "backend",
  llvm::cl::desc("Use specified backend"),
  llvm::cl::value_desc("backend"),
  llvm::cl::cat(OptionCategory));
llvm::cl::alias OptionBackendA(
    "b",
    llvm::cl::desc("Alias for -backend"),
    llvm::cl::aliasopt(OptionBackend));

llvm::cl::opt<std::string> OptionInputFilename(
  llvm::cl::Positional,
  llvm::cl::desc("Optional input file.  If none specified, run the interpreter"),
  llvm::cl::value_desc("input file"));

//==============================================================================
// Main driver code.
//==============================================================================
int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);
  
  // get arguments
  cl::ParseCommandLineOptions(argc, argv);


  // santity checks
  Contra Interp;
  Interp.setInteractive( OptionInputFilename.empty() );
  Interp.setVerbose( OptionVerbose );
  Interp.setDebug( OptionDebug );
  Interp.setOverwrite( OptionForce );
  Interp.setOptimized( OptionOptimizationLevel >= O1 );
  if (!OptionDumpIR.empty()) Interp.setDumpIR(OptionDumpIR);
  if (!OptionDumpDot.empty()) Interp.setDumpDot(OptionDumpDot);
  if (!OptionBackend.empty()) Interp.setBackend(OptionBackend);

  // if we are not interactive and compiling, open a file
  std::string source_filename;
  std::string output_filename;

  if (!Interp.isInteractive()) {
    
    source_filename = OptionInputFilename;
    if (Interp.isVerbose())
      std::cout << "Reading source file:" << source_filename << std::endl;
    
    if (OptionCompile) {
      if (!OptionOutput.empty()) {
        output_filename = OptionOutput;
      }
      else {
        auto source_extension = file_extension(source_filename);
        if ( source_extension == "cta" )
          output_filename = remove_extension(source_filename);
        else
          output_filename = source_filename;
        output_filename += ".o";
      }
      Interp.setCompile( output_filename );
    } // compile

  } // interactive

  // initialize llvm
  startLLVM();

  // install tokens
  Tokens::setup();

  // create the parser
  Interp.setup(source_filename);


  // Run the main "interpreter loop" now.
  Interp.mainLoop();
  
  MPI_Finalize();

  return 0;

}
