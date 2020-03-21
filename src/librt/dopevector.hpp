#ifndef RTLIB_DOPEVECTOR_HPP
#define RTLIB_DOPEVECTOR_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

#include "config.hpp"

#include <memory>
#include <string>

extern "C" {

/// forward declaration
struct dopevector_t;

/// memory allocation
DLLEXPORT dopevector_t allocate(int_t size);

/// memory deallocation
DLLEXPORT void deallocate(dopevector_t dv);

} // extern

namespace contra {
class FunctionDef;
}

namespace librt {

struct Allocate {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::shared_ptr<contra::FunctionDef> check() { return nullptr; };
};

struct DeAllocate {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::shared_ptr<contra::FunctionDef> check() { return nullptr; };
};

} // namespace


#endif // RTLIB_DOPEVECTOR_HPP
