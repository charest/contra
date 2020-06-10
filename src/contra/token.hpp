#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

#include <string>
#include <map>

namespace contra {

//==============================================================================
// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
//==============================================================================
enum Token {

  // commands
  tok_def = -100,

  // primary
  tok_identifier,

  // control
  tok_if,
  tok_elif,
  tok_else,

  // loops
  tok_for,
  tok_foreach,
  tok_range,

  tok_use,

  // operators
  tok_binary,
  tok_unary,

  // numbers / strings
  tok_true,
  tok_false,
  tok_char_literal,
  tok_int_literal,
  tok_real_literal,
  tok_string_literal,

  // functions
  tok_function,
  tok_end,
  tok_return,
  tok_task,
  tok_lambda = '@',

  // special binary
  tok_eq,
  tok_ne,
  tok_le,
  tok_ge,
  tok_asgmt_add,
  tok_asgmt_sub,
  tok_asgmt_mul,
  tok_asgmt_div,
  
  // file seperators
  tok_eof = -1,
  tok_sep = ';',
  
  // binary
  tok_asgmt = '=',
  tok_lt = '<',
  tok_gt = '>',
  tok_add = '+',
  tok_sub = '-',
  tok_mul = '*',
  tok_div = '/',
  tok_mod = '%'
};

//==============================================================================
// Helper class to return search result
//==============================================================================
struct TokenResult {
  bool found = false;
  int token = 0;
};

//==============================================================================
// Struct that contains all installed tokens
//==============================================================================
class Tokens {

  using map_type = std::map<int, std::string>;
  using reverse_map_type = std::map<std::string, int>;
  
  // Token map
  static map_type TokenMap;

  // Reserved keyword map
  static reverse_map_type KeywordMap;

public:

  // setup tokens
  static void setup();

  // Get a tokens name
  static std::string getName(int Tok);

  // get a token from its name
  static TokenResult getTok(const std::string & Name);
};

} // namespace

#endif // CONTRA_TOKEN_HPP
