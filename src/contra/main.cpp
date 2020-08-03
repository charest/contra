#include "contra.hpp"
#include "precedence.hpp"
#include "utils/string_utils.hpp"

#include "librt/librt.hpp"

#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <iostream>
#include <memory>

using namespace contra;
using namespace utils;
using namespace llvm;

//==============================================================================
// Options
//==============================================================================
static cl::OptionCategory OptionCategory("Contra Options");

cl::opt<bool> OptionVerbose(
    "verbose",
    cl::desc("Print extra information"),
    cl::cat(OptionCategory));
cl::alias OptionVerboseA(
    "v",
    cl::desc("Alias for -verbose"),
    cl::aliasopt(OptionVerbose));

cl::opt<bool> OptionForce(
    "force",
    cl::desc("Force overwriting of existing files"),
    cl::cat(OptionCategory));
cl::alias OptionForceA(
    "f",
    cl::desc("Alias for -force"),
    cl::aliasopt(OptionForce));

cl::opt<bool> OptionCompile(
    "c",
    cl::desc("Compile the source files"),
    cl::cat(OptionCategory));

cl::opt<std::string> OptionOutput(
    "o",
    cl::desc("Place compiled output in <filename>"),
    cl::value_desc("filename"),
    cl::cat(OptionCategory));

cl::opt<bool> OptionDebug(
    "g",
    cl::desc("Produce debugging information"),
    cl::cat(OptionCategory));

enum OptLevel {
  O0, O1, O2, O3
};

cl::opt<OptLevel> OptionOptimizationLevel(
  cl::desc("Choose optimization level:"),
  cl::values(
    clEnumVal(O0 , "No optimizations"),
    clEnumVal(O1, "Enable trivial optimizations"),
    clEnumVal(O2, "Enable default optimizations"),
    clEnumVal(O3, "Enable expensive optimizations")),
  cl::init(O0),
  cl::cat(OptionCategory));

cl::opt<std::string> OptionDumpIR(
  "dump-ir",
  cl::desc("Dump LLVM IR to <filename>"),
  cl::value_desc("filename"),
  cl::cat(OptionCategory));

cl::opt<std::string> OptionDumpDot(
  "dump-dot",
  cl::desc("Dump AST in graphviz format to <filename>"),
  cl::value_desc("filename"),
  cl::cat(OptionCategory));

cl::opt<std::string> OptionBackend(
  "backend",
  cl::desc("Use specified backend"),
  cl::value_desc("backend"),
  cl::cat(OptionCategory));
cl::alias OptionBackendA(
    "b",
    cl::desc("Alias for -backend"),
    cl::aliasopt(OptionBackend));

cl::opt<std::string> OptionInputFilename(
  cl::Positional,
  cl::desc("Optional input file.  If none specified, run the interpreter"),
  cl::value_desc("input file"));


//==============================================================================
// Main driver code.
//==============================================================================
int main(int argc, char** argv) {

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

  return 0;

}
