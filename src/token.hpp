#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

namespace contra {

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
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
  tok_in = -10
};

// operators
enum Operator {

  // binary
  op_lt = '<',
  op_add = '+',
  op_sub = '-',
  op_mul = '*',
  op_div = '/'

};

} // namespace

#endif // CONTRA_TOKEN_HPP
