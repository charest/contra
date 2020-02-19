#include "args.hpp"
#include "contra.hpp"
#include "llvm.hpp"
#include "string_utils.hpp"

#include <fstream>
#include <iostream>
#include <memory>

//==============================================================================
// "Library" functions that can be "extern'd" from user code.
//==============================================================================

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

using namespace contra;


//==============================================================================
// Main driver code.
//==============================================================================

int main(int argc, char** argv) {

  // get arguments
  std::map<std::string, std::string> args;
  auto res = process_arguments( argc, argv, args );  
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
  llvm_start();

  // create the parser
  std::unique_ptr<Parser> TheParser;
  if (!source_filename.empty())
    TheParser = std::make_unique<Parser>(source_filename);
  else
    TheParser = std::make_unique<Parser>();

  // Prime the first token.
  if (is_interactive) std::cerr << "ready> ";
  TheParser->getNextToken();

  // create the JIT and Code generator
  CodeGen TheCG(is_debug);

  // Run the main "interpreter loop" now.
  auto ret = MainLoop(*TheParser, TheCG, is_interactive);
  if (ret) return ret;

  // Finalize whatever needs to be
  TheCG.finalize();

  // Print out all of the generated code.
  //TheCG.TheModule->print(llvm::errs(), nullptr);

  // pile if necessary
  if (do_compile)
    res = llvm_compile( *TheCG.TheModule, output_filename );

  return res;

}
