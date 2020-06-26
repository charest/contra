#ifndef CONTRA_REDUCEINFO_HPP
#define CONTRA_REDUCEINFO_HPP

#include "llvm/IR/IRBuilder.h"

#include <string>

namespace contra {

//==============================================================================
// Reduction info
//==============================================================================
class AbstractReduceInfo {
public:

  AbstractReduceInfo() = default;
  virtual ~AbstractReduceInfo() {}
  
};

} // namespace

#endif // CONTRA_REDUCE_HPP
