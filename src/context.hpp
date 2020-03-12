#ifndef CONTRA_CONTEXT_HPP
#define CONTRA_CONTEXT_HPP

#include <memory>

namespace contra {

class Symbol;

struct Context {

  static std::shared_ptr<Symbol> I64Symbol;
  static std::shared_ptr<Symbol> F64Symbol;
  static std::shared_ptr<Symbol> StrSymbol;
  static std::shared_ptr<Symbol> BoolSymbol;
  static std::shared_ptr<Symbol> VoidSymbol;

};

}

#endif // CONTRA_CONTEXT_HPP
