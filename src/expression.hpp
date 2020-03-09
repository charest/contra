#ifndef CONTRA_EXPRESSION_HPP
#define CONTRA_EXPRESSION_HPP

#include "sourceloc.hpp"

namespace contra {

class ExprAST;

//==============================================================================
/// ExprBlock - List of expressions that form a block 
//==============================================================================

struct ExprLocPair {
  SourceLocation Loc;
  std::unique_ptr<ExprAST> Expr;
};
using ExprLocPairList = std::list< ExprLocPair >;

inline
void addExpr(ExprLocPairList & l, SourceLocation sl, std::unique_ptr<ExprAST> e)
{ l.emplace_back( ExprLocPair{sl, std::move(e) } ); }


using ExprBlock = std::vector< std::unique_ptr<ExprAST> >;
using ExprBlockList = std::list<ExprBlock>;

inline auto createBlock( ExprBlockList & list)
{ return list.emplace( list.end(), ExprBlock{} ); }

} // namespace

#endif // CONTRA_EXPRESSION_HPP
