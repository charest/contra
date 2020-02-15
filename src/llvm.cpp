#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace contra {

void llvm_start() {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

}

} // namespace
