#ifndef CONTRA_ARRAY_HPP
#define CONTRA_ARRAY_HPP

namespace llvm {
class AllocaInst;
class Value;
}

namespace contra {

struct ArrayType {
  llvm::AllocaInst* Alloca = nullptr;
  llvm::Value* Data = nullptr;
  llvm::Value* Size = nullptr;
};

} // namespace

#endif // CONTRA_ARRAY_HPP
