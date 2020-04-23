#ifndef RTLIB_RANGE_HPP
#define RTLIB_RANGE_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

#include "config.hpp"

#include <memory>
#include <string>

extern "C" {

/// simple range type
struct range_t;

} // extern

namespace contra {
class FunctionDef;
}

namespace librt {

struct Range {
  static llvm::Type* RangeType;
  static const std::string Name;
  static void setup(llvm::LLVMContext &);
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &) {return nullptr;}
  static std::unique_ptr<contra::FunctionDef> check() {return nullptr;}
};

} // namespace


#endif // RTLIB_RANGE_HPP
