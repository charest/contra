#include "args.hpp"
#include "contra.hpp"
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
  Contra Interp;
  Interp.setInteractive( args.count("__positional") == 0 );
  Interp.setVerbose( args.count("v") );
  Interp.setDebug( args.count("g") );
  Interp.setOverwrite( args.count("f") );
  Interp.setOptimized( args.count("O") );
  if (args.count("i")) Interp.setDumpIR(args.at("i"));
  if (args.count("d")) Interp.setDumpDot(args.at("d"));

  // if we are not interactive and compiling, open a file
  std::string source_filename;
  std::string output_filename;

  if (!Interp.isInteractive()) {
    
    source_filename = split( args.at("__positional"), ';' )[0];
    if (Interp.isVerbose())
      std::cout << "Reading source file:" << source_filename << std::endl;
    
    if (args.count("c")) {
      if (args.count("o")) {
        output_filename = args.at("o");
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
