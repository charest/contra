#include "ast.hpp"
#include "token.hpp"

#include <vector>

namespace contra {

//==============================================================================
// Make sext tokens
//==============================================================================
Tokens make_sext_tokens() {
  auto toks = make_contra_tokens();
  
  std::vector<int> ast_toks = {
    ast_unk,
    ast_fn_def,
    ast_fn_call,
    ast_fn_anon,
    ast_fn_args,
    ast_fn_body,
    ast_var,
    ast_arr_index,
    ast_if,
    ast_if_cond,
    ast_if_body,
    ast_elif_cond,
    ast_elif_body,
    ast_else_body,
    ast_for,
    ast_for_body,
    ast_foreach,
    ast_break,
    ast_lit_real,
    ast_lit_int,
    ast_lit_string,
    ast_arr,
    ast_reduce,
    ast_unary,
    ast_binop,
    ast_range,
    ast_assign,
    ast_expr_list,
    ast_use,
    ast_reduce,
    ast_reduce_op,
    ast_return,
  };
  for (auto tok : ast_toks)
    toks.keywords.add(tok, ast_to_string(tok) );
    
  return toks;
}

} // contra
