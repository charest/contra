#include "token.hpp"

namespace contra {

// Initializers
Tokens::map_type Tokens::TokenMap = {};
Tokens::reverse_map_type Tokens::KeywordMap = {};

//==============================================================================
// Install the tokens
//==============================================================================
void Tokens::setup() {

  // add non-keywords here
  TokenMap = {
    { tok_eof, "eof" },
    { tok_identifier, "identifier" },
    { tok_int_number, "integer_number" },
    { tok_real_number, "real_number" }
  };

  // add keywords here
  std::map<int, std::string> Keywords = {
    { tok_binary, "binary" },
    { tok_by, "by" },
    { tok_def, "def" },
    { tok_do, "do" },
    { tok_elif, "elif" },
    { tok_else, "else" },
    { tok_end, "end" },
    { tok_extern, "extern" },
    { tok_for, "for" },
    { tok_function, "function" },
    { tok_if, "if" },
    { tok_in, "in" },
    { tok_int, "i64" },
    { tok_real, "f64" },
    { tok_return, "return" },
    { tok_string, "string" },
    { tok_then, "then" },
    { tok_to, "to" },
    { tok_unary, "unary" },
    { tok_until, "until" },
    { tok_var, "var" }
  };

  // create keyword list
  KeywordMap.clear();
  for ( const auto & key_pair : Keywords )
    KeywordMap.emplace( key_pair.second, key_pair.first );

  // insert keywords into full token map
  TokenMap.insert( Keywords.begin(), Keywords.end() );
}
  
//==============================================================================
// Get a tokens name
//==============================================================================
std::string Tokens::getName(int Tok) {
  auto it = TokenMap.find(Tok);
  if (it != TokenMap.end()) return it->second;
  return std::string(1, (char)Tok);
}
  
//==============================================================================
// get a token from its name
//==============================================================================
TokenResult Tokens::getTok(const std::string & Name)
{
  auto it = KeywordMap.find(Name);
  if (it != KeywordMap.end())
    return {true, it->second };
  else 
    return {false, 0};
}

} // namespace
