#ifndef CONTRA_TOKEN_HPP
#define CONTRA_TOKEN_HPP

#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <unordered_map>

namespace contra {
  
#define TOKEN_NOT_FOUND -1000


//==============================================================================
// Helper class to return search result
//==============================================================================
struct TokenResult {
  bool found = false;
  int token = 0;
};

struct token_set_t {
  std::unordered_set<int> tokens;

  void add(int tok)
  { tokens.insert(tok); }

  int find(int tok) const
  { return tokens.count(tok) ? tok : TOKEN_NOT_FOUND; }
};


struct token_map_t {
  std::unordered_map<int, std::string> enum_to_str;
  std::unordered_map<std::string, int> str_to_enum;

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
    return TOKEN_NOT_FOUND;
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
  
  // Specials
  int eof = -1;
  int identifier = 0;
  int real_literal = 1;
  int int_literal = 2;
  int string_literal = 3;
  int comment = '#';
  int quote = '\"';

  // token data
  token_set_t exact_symbols;
  token_map_t inexact_symbols;
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


} // namespace

#endif // CONTRA_TOKEN_HPP
