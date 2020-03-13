
#include "context.hpp"
#include "symbols.hpp"

namespace contra {

std::shared_ptr<TypeDef> Context::I64Type = std::make_shared<BuiltInTypeDef>("i64", true);
std::shared_ptr<TypeDef> Context::F64Type = std::make_shared<BuiltInTypeDef>("f64", true);
std::shared_ptr<TypeDef> Context::StrType = std::make_shared<BuiltInTypeDef>("string");
std::shared_ptr<TypeDef> Context::BoolType = std::make_shared<BuiltInTypeDef>("bool");
std::shared_ptr<TypeDef> Context::VoidType = std::make_shared<BuiltInTypeDef>("void");

}
