#ifndef RTLIB_PRINT_HPP
#define RTLIB_PRINT_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

#include <string>
#include <memory>

extern "C" {

/// generic c print statement
DLLEXPORT void print(const char *format, ...);

} // extern

namespace contra {
class FunctionDef;
}

namespace librt {

struct Print {
  static const std::string Name;
  static void setup(llvm::LLVMContext &) {};
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::unique_ptr<contra::FunctionDef> check();

};

} // namespace


#endif // RTLIB_PRINT_HPP
