#include "toks.hpp"
#include "token.hpp"
#include "precedence.hpp"

namespace contra {

//==============================================================================
// Make contra tokens
//==============================================================================
Tokens make_contra_tokens() {
  Tokens toks;

  toks.exact_symbols.add( tok_comment );
  toks.exact_symbols.add( tok_sep );
  toks.exact_symbols.add( tok_comma );
  toks.exact_symbols.add( tok_colon );
  toks.exact_symbols.add( tok_asgmt );
  toks.exact_symbols.add( tok_lt );
  toks.exact_symbols.add( tok_gt );
  toks.exact_symbols.add( tok_add );
  toks.exact_symbols.add( tok_sub );
  toks.exact_symbols.add( tok_mul );
  toks.exact_symbols.add( tok_div );
  toks.exact_symbols.add( tok_mod );
  toks.exact_symbols.add( tok_lparens );
  toks.exact_symbols.add( tok_rparens );
  toks.exact_symbols.add( tok_lbrack );
  toks.exact_symbols.add( tok_rbrack );
  toks.exact_symbols.add( tok_lbrace );
  toks.exact_symbols.add( tok_rbrace );
    
  toks.inexact_symbols.add( tok_eq, "==" );
  toks.inexact_symbols.add( tok_ne, "!=" );
  toks.inexact_symbols.add( tok_le, "<=" );
  toks.inexact_symbols.add( tok_ge, ">=" );
  toks.inexact_symbols.add( tok_asgmt_add, "+=" );
  toks.inexact_symbols.add( tok_asgmt_sub, "-=" );
  toks.inexact_symbols.add( tok_asgmt_mul, "*=" );
  toks.inexact_symbols.add( tok_asgmt_div, "/=" );
  
  toks.keywords.add( tok_if, "if" );
  toks.keywords.add( tok_elif, "elif" );
  toks.keywords.add( tok_else, "else" );
  toks.keywords.add( tok_for, "for" );
  toks.keywords.add( tok_foreach, "foreach" );
  toks.keywords.add( tok_break, "break" );
  toks.keywords.add( tok_reduce, "reduce" );
  toks.keywords.add( tok_use, "use" );
  toks.keywords.add( tok_true, "true" );
  toks.keywords.add( tok_false, "false" );
  toks.keywords.add( tok_function, "fn" );
  toks.keywords.add( tok_return, "return" );
  toks.keywords.add( tok_task, "tsk" );
  
  toks.types.add( tok_i64, "i64" );
  toks.types.add( tok_f64, "f64" );
  
  toks.tags.add( tok_eof, "eof" );
  toks.tags.add( tok_ident, "identifier" );
  toks.tags.add( tok_char_lit, "char_lit" );
  toks.tags.add( tok_int_lit, "integer_lit" );
  toks.tags.add( tok_real_lit, "real_lit" );
  toks.tags.add( tok_string_lit, "string_lit" ); 
  
  toks.eof = tok_eof;
  toks.identifier = tok_ident;
  toks.real_literal = tok_real_lit;
  toks.int_literal = tok_int_lit;
  toks.string_literal = tok_string_lit;
  toks.comment = tok_comment;
  toks.quote = '\"';
    
  return toks;
}

//==============================================================================
// Make contra precedence
//==============================================================================
BinopPrecedence make_contra_precedence() {
  BinopPrecedence p;

  // Install standard binary operators.
  // 1 is lowest precedence.
  //Precedence_[tok_asgmt] = 2;
  p.add( tok_eq , 5 );
  p.add( tok_ne , 5 );
  p.add( tok_lt , 10);
  p.add( tok_le , 10);
  p.add( tok_gt , 10);
  p.add( tok_ge , 10);
  p.add( tok_add, 20);
  p.add( tok_sub, 20);
  p.add( tok_mul, 40);
  p.add( tok_div, 40);
  p.add( tok_mod, 40);
  
  int prec{1};
  //p.binary_left[tok_or] = prec;
  //++prec;
  //p.binary_left[tok_and] = prec;
  //++prec;
  p.binary_left[tok_eq] = prec;
  p.binary_left[tok_ne] = prec;
  ++prec;
  p.binary_left[tok_lt] = prec;
  p.binary_left[tok_le] = prec;
  p.binary_left[tok_gt] = prec;
  p.binary_left[tok_ge] = prec;
  ++prec;
  p.binary_left[tok_add] = prec;
  p.binary_left[tok_sub] = prec;
  ++prec;
  p.binary_left[tok_mul] = prec;
  p.binary_left[tok_div] = prec;
  p.binary_left[tok_mod] = prec;
  //++prec;
  //p.binary_right[tok_pow] = prec;
  ++prec;
  p.unary[tok_add] = prec;
  p.unary[tok_sub] = prec;
  //p.unary[tok_not] = prec;
  // highest.
    
  return p;
}


} // namespace
