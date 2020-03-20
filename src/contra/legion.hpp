#ifndef CONTRA_LEGION_HPP
#define CONTRA_LEGION_HPP

#include "tasking.hpp"

namespace contra {

class LegionTasker : public AbstractTasker {
public:
 
  LegionTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext) :
    AbstractTasker(TheBuilder, TheContext)
  {}

  virtual llvm::Function* wrap(llvm::Module &, const std::string &,
      llvm::Function*) const override;
  
  virtual ~LegionTasker() = default;
};

} // namepsace

#endif // LIBRT_LEGION_HPP
