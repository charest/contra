#ifndef CONTRA_LEAF_HPP
#define CONTRA_LEAF_HPP

#include "config.hpp"
#include "recursive.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
template< typename T, typename... Args >
class Matcher : public RecursiveAstVisiter {

  static constexpr auto NumMembers_ = sizeof...(Args);

  bool DoRecurse_ = false;
  T* Head_ = nullptr;

  using MemberTuple = std::tuple<Args*...>;
  MemberTuple Members_;

  std::vector<NodeAST*> Matches;

  template<
    std::size_t I,
    typename MemberType = typename std::tuple_element<0, MemberTuple>::type,
    bool IsVoid = std::is_same<MemberType, void>::value
    >
  bool matchMember(NodeAST* Expr)
  {
    if (IsVoid) return true;
    return dynamic_cast<MemberType>(Expr);
  }
  
public:

  auto runVisitor(NodeAST&e)
  {
    e.accept(*this);
    return Matches;
  }
  
  bool preVisit(BinaryExprAST& e) override
  {
    if (NumMembers_ == 2) {
      if (matchMember<0>(e.getLeftExpr()) && matchMember<1>(e.getRightExpr())) {
        Matches.emplace_back(&e);
        if (!DoRecurse_) return true;
      }
    }
    return true;
  }

};

////////////////////////////////////////////////////////////////////////////////
/// The mathing function
////////////////////////////////////////////////////////////////////////////////
template< typename T, typename... Args >
std::vector<NodeAST*>
match(NodeAST* root, bool recurse = true)
{
  Matcher<T, Args...> M;
  return M.runVisitor(*root);
}

} // namespace

#endif // CONTRA_LEAF_HPP
