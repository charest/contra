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
  
  virtual void preregister(llvm::Module &, const std::string &, const TaskInfo &) const override;
  
  virtual void set_top(llvm::Module &, int) const override;
  
  virtual llvm::Value* start(llvm::Module &, int, char **) const override;
  
  virtual ~LegionTasker() = default;
};

} // namepsace

#endif // LIBRT_LEGION_HPP
