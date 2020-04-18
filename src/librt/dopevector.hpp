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
DLLEXPORT void allocate(int_t size, int_t data_size, dopevector_t * dv);

/// memory deallocation
DLLEXPORT void deallocate(dopevector_t * dv);

/// memory deallocation
DLLEXPORT void copy(dopevector_t * src, dopevector_t * tgt);

} // extern

namespace contra {
class FunctionDef;
}

namespace librt {

struct DopeVector {
  static llvm::Type* DopeVectorType;
  static void setup(llvm::LLVMContext &);
};

struct Allocate : public DopeVector {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::unique_ptr<contra::FunctionDef> check();
};

struct DeAllocate : public DopeVector {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::unique_ptr<contra::FunctionDef> check();
};

struct Copy : public DopeVector {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::unique_ptr<contra::FunctionDef> check();
};

} // namespace


#endif // RTLIB_DOPEVECTOR_HPP
