#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

#include <iostream>
#include <string>
#include <map>

namespace contra {

//==============================================================================
// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
//==============================================================================
enum Token {

  tok_not_found = -1000,
  
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
  tok_identifier,
  
  // operators
  tok_binary, // TODO delete
  tok_unary, // TODO delete
  
  // numbers / strings
  tok_char_literal,
  tok_int_literal,
  tok_real_literal,
  tok_string_literal,
  
  // file seperators
  tok_eof = -1

};

//==============================================================================
// Helper class to return search result
//==============================================================================
struct TokenResult {
  bool found = false;
  int token = 0;
};

struct token_map_t {
  std::map<int, std::string> enum_to_str;
  std::map<std::string, int> str_to_enum;

  void add(int tok, const std::string & str)
  {
    enum_to_str[tok] = str;
    str_to_enum[str] = tok;
  }
  
  void add(int tok)
  {
    enum_to_str[tok] = tok;
    str_to_enum[std::string(1,tok)] = tok;
  }

  std::string find(int tok) const
  {
    auto it = enum_to_str.find(tok);
    if (it != enum_to_str.end()) return it->second;
    return {};
  }
  
  int find(const std::string & str) const
  {
    auto it = str_to_enum.find(str);
    if (it != str_to_enum.end()) return it->second;
    return tok_not_found;
  }
};

//==============================================================================
// Struct that contains all installed tokens
//==============================================================================
struct Tokens {

  using map_type = std::map<int, std::string>;
  using reverse_map_type = std::map<std::string, int>;
  
  // Token map
  static map_type TokenMap;

  // Reserved keyword map
  static reverse_map_type KeywordToToken;
  static reverse_map_type TypeKeywordToToken;
  static map_type TypeTokenToKeyword;

  // token data
  token_map_t one_char;
  token_map_t multi_char;
  token_map_t keywords;
  token_map_t types;
  token_map_t tags;

  // search api
  std::string findInAll(int search) const;
  int findInAll(const std::string & search) const;

  // setup tokens
  static void setup();

  // Get a tokens name
  static std::string getName(int Tok);

  // get a token from its name
  static TokenResult getTok(const std::string & Name);

  static bool isType(int tok);
};

/// Main token builder 
Tokens make_contra_tokens();

} // namespace

#endif // CONTRA_TOKEN_HPP
