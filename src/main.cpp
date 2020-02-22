#include "args.hpp"
#include "contra.hpp"
#include "llvm.hpp"
#include "string_utils.hpp"

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
  bool is_interactive = args.count("__positional") == 0;
  bool do_compile = args.count("c");
  bool is_verbose = args.count("v");
  bool has_output = args.count("o");
  bool is_debug = args.count("g");


  // if we are not interactive and compiling, open a file
  std::string source_filename;
  std::string output_filename;

  if (!is_interactive) {
    
    source_filename = split( args.at("__positional"), ';' )[0];
    if (is_verbose) std::cout << "Reading source file:" << source_filename << std::endl;
    
    if (do_compile) {
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

  // create the parser
  std::unique_ptr<Parser> TheParser;
  if (!source_filename.empty())
    TheParser = std::make_unique<Parser>(source_filename, is_verbose);
  else
    TheParser = std::make_unique<Parser>(is_verbose);

  // create the JIT and Code generator
  CodeGen TheCG(is_debug);

  // Run the main "interpreter loop" now.
  mainLoop(*TheParser, TheCG, is_interactive, is_verbose);

  // Finalize whatever needs to be
  TheCG.finalize();

  // Print out all of the generated code.
  //TheCG.TheModule->print(llvm::errs(), nullptr);

  // pile if necessary
  if (do_compile)
    compileLLVM( *TheCG.TheModule, output_filename );

  return 0;

}
