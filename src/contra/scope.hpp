#ifndef CONTRA_SCOPE_HPP
#define CONTRA_SCOPE_HPP

#include "config.hpp"

namespace contra {

class Scoper {
public:

  using value_type = int_t;

  static constexpr value_type GlobalScope = 0;

  virtual ~Scoper() = default;
  
  auto getScope() const { return Scope_; };

  bool isGlobalScope() const { return Scope_ == GlobalScope; }

  virtual void resetScope(value_type Scope)
  { Scope_ = Scope; }
  
  virtual value_type createScope()
  { 
    Scope_++;
    return Scope_;
  }
  

private:

  value_type Scope_ = GlobalScope;

};


} // namespace

#endif // CONTRA_ARRAY_HPP
