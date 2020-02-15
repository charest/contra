#include "contra.hpp"
#include "llvm.hpp"

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

int main() {

  llvm_start();

  // create the parser
  Parser TheParser;

  // Prime the first token.
  std::cerr << "ready> ";
  TheParser.getNextToken();

  // create the JIT and Code generator
  ContraJIT TheJIT;
  CodeGen TheCG;

  InitializeModuleAndPassManager(TheCG, TheJIT);

  // Run the main "interpreter loop" now.
  MainLoop(TheParser, TheCG, TheJIT);

  return 0;
}
