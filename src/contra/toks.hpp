#ifndef CONTRA_CONTRA_TOKS_HPP
#define CONTRA_CONTRA_TOKS_HPP

namespace contra {

struct BinopPrecedence;
struct Tokens;
  
//==============================================================================
// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
//==============================================================================
enum Token {

  //--- ONE CHAR, i.e. single character symbols

  // grammar
  tok_comment = '#',
  tok_sep = ';',
  tok_comma = ',',
  tok_colon = ':',
  
  // binary
  tok_asgmt = '=',
  tok_lt = '<',
  tok_gt = '>',
  tok_add = '+',
  tok_sub = '-',
  tok_mul = '*',
  tok_div = '/',
  tok_mod = '%',

  // brackets
  tok_lparens = '(',
  tok_rparens = ')',
  tok_lbrack  = '[',
  tok_rbrack  = ']',
  tok_lbrace  = '{',
  tok_rbrace  = '}',

  
  //--- MULTI-CHAR, i.e. multi-character symbols
  
  // special binary
  tok_eq = 256,
  tok_ne,
  tok_le,
  tok_ge,
  tok_asgmt_add,
  tok_asgmt_sub,
  tok_asgmt_mul,
  tok_asgmt_div,
  
  //--- KEYWORDS, i.e. maps to an alphanumeric keyword

  // control
  tok_if,
  tok_elif,
  tok_else,

  // loops
  tok_for,
  tok_foreach,
  tok_break,

  tok_reduce,
  tok_use,

  // booleans
  tok_true,
  tok_false,

  // functions
  tok_function,
  tok_return,
  tok_task,
  
  //--- TYPES, i.e. type keywords

  tok_i64,
  tok_f64,


  //--- TAGS, i.e. doesnt map to any text
  
  // primary
  tok_ident,
  
  // operators
  tok_binary, // TODO delete
  tok_unary, // TODO delete
  
  // numbers / strings
  tok_char_lit,
  tok_int_lit,
  tok_real_lit,
  tok_string_lit,
  
  // file seperators
  tok_eof,

  // total size
  tok_last,

};

/// Main token builder 
Tokens make_contra_tokens();

/// Main precedence builder
BinopPrecedence make_contra_precedence();


} // namespace

#endif
