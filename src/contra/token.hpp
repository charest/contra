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
  tok_until,

  // operators
  tok_binary,
  tok_unary,

  // variables
  tok_var,

  // numbers / strings
  tok_string,
  tok_int_number,
  tok_real_number,

  // functions
  tok_function,
  tok_end,
  tok_return,
  tok_task,

  // special binary
  tok_eq,
  tok_ne,
  tok_le,
  tok_ge,
  
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
