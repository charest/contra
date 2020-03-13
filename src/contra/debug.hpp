#ifndef CONTRA_DEBUG_HPP
#define CONTRA_DEBUG_HPP

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include <vector>

namespace contra {

class NodeAST;

struct DebugInfo {
  llvm::DICompileUnit *TheCU = nullptr;
  llvm::DIType *DblTy = nullptr;
  std::vector<llvm::DIScope *> LexicalBlocks;

  llvm::DIType *getDoubleTy(llvm::DIBuilder & DBuilder);
  void emitLocation(llvm::IRBuilder<> & Builder, NodeAST *AST);
};

} // namespace

#endif // CONTRA_DEBUG_HPP
