#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

#include <string>

namespace contra {

//==============================================================================
// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
//==============================================================================
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_in = -10,

  // operators
  tok_binary = -11,
  tok_unary = -12,

  // variables
  tok_var = -13,

  // new
  tok_function = -14,
  tok_end = -15,
  tok_return = -16,
  tok_to = -17,
  tok_by = -18,
  tok_do = -19,
  tok_string = -20,
  tok_elif = -21,

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
  tok_return,
  tok_then,
  tok_to,
  tok_unary,
  tok_var
};

const int num_keywords = sizeof(tok_keywords) / sizeof(Token);

} // namespace

#endif // CONTRA_TOKEN_HPP
