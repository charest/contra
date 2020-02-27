#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

#include <string>

namespace contra {

//==============================================================================
// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
//==============================================================================
enum Token {

  // commands
  tok_def = -100,
  tok_extern,

  // primary
  tok_identifier,

  // control
  tok_if,
  tok_then,
  tok_elif,
  tok_else,

  // loops
  tok_for,
  tok_in,
  tok_do,
  tok_to,
  tok_by,

  // operators
  tok_binary,
  tok_unary,

  // variables
  tok_var,
  tok_int,
  tok_real,
  tok_string,

  // numbers
  tok_int_number,
  tok_real_number,

  // functions
  tok_function,
  tok_end,
  tok_return,
  
  
  // file seperators
  tok_eof = -1,
  tok_sep = ';',
  
  // binary
  tok_eq = '=',
  tok_lt = '<',
  tok_add = '+',
  tok_sub = '-',
  tok_mul = '*',
  tok_div = '/'
};

// Get a tokens name
std::string getTokName(int Tok);
 
const Token tok_keywords[] = {
  tok_binary,
  tok_by,
  tok_def,
  tok_do,
  tok_elif,
  tok_else,
  tok_end,
  tok_extern,
  tok_for,
  tok_function,
  tok_if,
  tok_in,
  tok_int,
  tok_real,
  tok_return,
  tok_then,
  tok_to,
  tok_unary,
  tok_var
};

const int num_keywords = sizeof(tok_keywords) / sizeof(Token);

} // namespace

#endif // CONTRA_TOKEN_HPP
