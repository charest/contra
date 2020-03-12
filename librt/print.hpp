#ifndef RTLIB_PRINT_HPP
#define RTLIB_PRINT_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

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
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::shared_ptr<contra::FunctionDef> check() { return nullptr; };

};

} // namespace


#endif // RTLIB_PRINT_HPP
