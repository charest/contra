#ifndef RTLIB_MATH_HPP
#define RTLIB_MATH_HPP

#include "llvm_forwards.hpp"

#include <string>
#include <memory>

namespace contra {
class FunctionDef;
}

namespace librt {

struct CSqrt {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::shared_ptr<contra::FunctionDef> check();
};

struct CAbs {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::shared_ptr<contra::FunctionDef> check();
};

struct CMax {
  static const std::string Name;
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::shared_ptr<contra::FunctionDef> check();
};

} // namespace


#endif // RTLIB_MATH_HPP
