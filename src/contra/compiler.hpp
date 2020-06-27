#ifndef CONTRA_COMPILER_HPP
#define CONTRA_COMPILER_HPP

#include "config.hpp"

#include <string>

namespace llvm {
class Module;
}

namespace contra {

void compile(llvm::Module &, const std::string &);

} // namespace

#endif // CONTRA_COMPILER_HPP
