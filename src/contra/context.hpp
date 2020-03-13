#ifndef CONTRA_CONTEXT_HPP
#define CONTRA_CONTEXT_HPP

#include <memory>

namespace contra {

class TypeDef;

struct Context {

  static std::shared_ptr<TypeDef> I64Type;
  static std::shared_ptr<TypeDef> F64Type;
  static std::shared_ptr<TypeDef> StrType;
  static std::shared_ptr<TypeDef> BoolType;
  static std::shared_ptr<TypeDef> VoidType;

};

}

#endif // CONTRA_CONTEXT_HPP
