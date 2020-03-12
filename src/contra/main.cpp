#include "args.hpp"
#include "contra.hpp"
#include "inputs.hpp"
#include "llvm.hpp"
#include "precedence.hpp"
#include "string_utils.hpp"

#include "librt/librt.hpp"

#include <fstream>
#include <iostream>
#include <memory>

//==============================================================================
// Main driver code.
//==============================================================================
using namespace contra;

int main(int argc, char** argv) {

  // get arguments
  std::map<std::string, std::string> args;
  auto res = processArguments( argc, argv, args );  
  if ( args.count("h") ) return 0;
  if ( res ) return res;

  // santity checks
  InputsType inp;
  inp.is_interactive = args.count("__positional") == 0;
  inp.do_compile = args.count("c");
  inp.is_verbose = args.count("v");
  inp.has_output = args.count("o");
  inp.is_debug = args.count("g");
  inp.is_optimized = args.count("O");
  inp.dump_ir = args.count("i");

  // if we are not interactive and compiling, open a file
  std::string source_filename;
  std::string output_filename;

  if (!inp.is_interactive) {
    
    source_filename = split( args.at("__positional"), ';' )[0];
    if (inp.is_verbose) std::cout << "Reading source file:" << source_filename << std::endl;
    
    if (inp.do_compile) {
      auto source_extension = file_extension(source_filename);
      if ( source_extension == "cta" )
        output_filename = remove_extension(source_filename);
      else
        output_filename = source_filename;
      output_filename += ".o";
    } // compile

  } // interactive

  // initialize llvm
  startLLVM();

  // install tokens
  Tokens::setup();

  // setup runtime
  librt::RunTimeLib::setup();

  // create the operator precedence
  auto ThePrecedence = std::make_shared<BinopPrecedence>();

  // create the parser
  std::unique_ptr<Parser> TheParser;
  if (!source_filename.empty())
    TheParser = std::make_unique<Parser>(ThePrecedence, source_filename);
  else
    TheParser = std::make_unique<Parser>(ThePrecedence);

  // create the JIT and Code generator
  CodeGen TheCG(ThePrecedence, inp.is_debug);

  // Run the main "interpreter loop" now.
  mainLoop(*TheParser, TheCG, inp);

  // Finalize whatever needs to be
  TheCG.finalize();

  // Print out all of the generated code.
  //TheCG.TheModule->print(llvm::errs(), nullptr);

  // pile if necessary
  if (inp.do_compile)
    compileLLVM( TheCG.getModule(), output_filename );

  return 0;

}
