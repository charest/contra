#include "symbols.hpp"

namespace contra {

constexpr Attributes TypeDef::Attr::None;
constexpr Attributes TypeDef::Attr::Number;
    
constexpr Attributes VariableType::Attr::None;
constexpr Attributes VariableType::Attr::Array;
constexpr Attributes VariableType::Attr::Future;
constexpr Attributes VariableType::Attr::Global;
constexpr Attributes VariableType::Attr::Range;

constexpr Attributes FunctionDef::Attr::None;
constexpr Attributes FunctionDef::Attr::Task;

} // namespace
