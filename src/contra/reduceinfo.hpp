#ifndef CONTRA_REDUCEINFO_HPP
#define CONTRA_REDUCEINFO_HPP

#include "llvm/IR/IRBuilder.h"

#include <string>

namespace contra {

//==============================================================================
// Reduction info
//==============================================================================
class ReduceInfo {
  int Id_ = -1;

  llvm::FunctionType* ApplyT_ = nullptr;
  llvm::FunctionType* FoldT_ = nullptr;
  llvm::FunctionType* InitT_ = nullptr;
  
  std::string ApplyN_;
  std::string FoldN_;
  std::string InitN_;

  std::size_t DataSize_ = 0;

public:

  ReduceInfo(
      int Id,
      llvm::Function* Apply,
      llvm::Function* Fold,
      llvm::Function* Init,
      std::size_t DataSize)
    : 
      Id_(Id),
      ApplyT_(Apply->getFunctionType()),
      FoldT_(Fold->getFunctionType()),
      InitT_(Init->getFunctionType()),
      ApplyN_(Apply->getName()),
      FoldN_(Fold->getName()),
      InitN_(Init->getName()),
      DataSize_(DataSize)
  {}

  auto getId() const { return Id_; }
  
  auto getApplyType() const { return ApplyT_; }
  const auto & getApplyName() const { return ApplyN_; }

  auto getFoldType() const { return FoldT_; }
  const auto & getFoldName() const { return FoldN_; }

  auto getInitType() const { return InitT_; }
  const auto & getInitName() const { return InitN_; }

  auto getDataSize() const { return DataSize_; }

};

} // namespace

#endif // CONTRA_REDUCE_HPP
