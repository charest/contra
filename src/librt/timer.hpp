#ifndef RTLIB_TIMER_HPP
#define RTLIB_TIMER_HPP

#include "dllexport.h"
#include "llvm_forwards.hpp"

#include "config.hpp"

#include <string>
#include <memory>

extern "C" {

/// get wall time
DLLEXPORT real_t timer(void);

} // extern

namespace contra {
class FunctionDef;
}

namespace librt {

struct Timer {
  static const std::string Name;
  static void setup(llvm::LLVMContext &) {};
  static llvm::Function *install(llvm::LLVMContext &, llvm::Module &);
  static std::unique_ptr<contra::FunctionDef> check();

};

} // namespace


#endif // RTLIB_TIMER_HPP
