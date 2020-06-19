#ifndef CONTRA_REDUCTIONS_HPP
#define CONTRA_REDUCTIONS_HPP

#include <map>

namespace contra {
  
enum class ReductionType {
  Add,
  Sub,
  Mult,
  Div,
  Min,
  Max,
  User
};

struct SupportedReductions {

  static std::map<std::string, ReductionType> Map;

  static ReductionType getType(const std::string & Name);

};

} // namespace

#endif // CONTRA_PRECEDENCE_HPP
