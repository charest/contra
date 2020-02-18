#include "ast.hpp"
#include "debug.hpp"

using namespace llvm;

namespace contra {

//==============================================================================
//==============================================================================
DIType *DebugInfo::getDoubleTy(DIBuilder & DBuilder) {
  if (DblTy)
    return DblTy;

  DblTy = DBuilder.createBasicType("double", 64, dwarf::DW_ATE_float);
  return DblTy;
}

//==============================================================================
// tell Builder whenever we’re at a new source location
//==============================================================================
void DebugInfo::emitLocation(IRBuilder<> & Builder, ExprAST *AST) {
  if (!AST)
    return Builder.SetCurrentDebugLocation(DebugLoc());
  DIScope *Scope;
  if (LexicalBlocks.empty())
    Scope = TheCU;
  else
    Scope = LexicalBlocks.back();
  Builder.SetCurrentDebugLocation(
      DebugLoc::get(AST->getLine(), AST->getCol(), Scope));
}

}
