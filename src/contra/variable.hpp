#ifndef CONTRA_VARIABLE_HPP
#define CONTRA_VARIABLE_HPP

namespace llvm {
class Type;
class Value;
}

namespace contra {

class VariableAlloca {
  llvm::Value* Alloca_ = nullptr;
  llvm::Type* Type_ = nullptr;
  llvm::Value* Size_ = nullptr;
  bool IsOwner_ = true;
public:
  VariableAlloca() = default;
  VariableAlloca(llvm::Value* Alloca, llvm::Type* Type, llvm::Value* Size = nullptr)
    : Alloca_(Alloca), Type_(Type), Size_(Size) {}
  auto getAlloca() const { return Alloca_; }
  void setAlloca(llvm::Value* Alloca) { Alloca_ = Alloca; }
  void setAlloca(llvm::Value* Alloca, bool IsOwner)
  {
    Alloca_ = Alloca;
    IsOwner_ = IsOwner;
  }
  auto getType() const { return Type_; }
  auto getSize() const { return Size_; }
  void setOwner(bool IsOwner=true) { IsOwner_=IsOwner; }
  auto isOwner() const { return IsOwner_; }
};

} // namespace

#endif // CONTRA_VARIABLE_HPP
