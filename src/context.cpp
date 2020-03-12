
#include "context.hpp"
#include "symbols.hpp"

namespace contra {

std::shared_ptr<Symbol> Context::I64Symbol = std::make_shared<BuiltInSymbol>("i64");
std::shared_ptr<Symbol> Context::F64Symbol = std::make_shared<BuiltInSymbol>("f64");
std::shared_ptr<Symbol> Context::StrSymbol = std::make_shared<BuiltInSymbol>("string");
std::shared_ptr<Symbol> Context::BoolSymbol = std::make_shared<BuiltInSymbol>("bool");
std::shared_ptr<Symbol> Context::VoidSymbol = std::make_shared<BuiltInSymbol>("void");

}
