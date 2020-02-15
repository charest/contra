#ifndef CONTRA_LLVM_HPP
#define CONTRA_LLVM_HPP

namespace llvm {
class Module;
}

/// start llvm
namespace contra {

void llvm_start();
int llvm_compile(llvm::Module &);

}


#endif //CONTRA_LLVM_HPP
