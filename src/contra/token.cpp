#include "token.hpp"
#include "toks.hpp"

namespace contra {

// Initializers
Tokens::map_type Tokens::TokenMap = {};
Tokens::reverse_map_type Tokens::KeywordToToken = {};
Tokens::reverse_map_type Tokens::TypeKeywordToToken = {};
Tokens::map_type Tokens::TypeTokenToKeyword = {};
  
std::string Tokens::findInAll(int search) const
{
  auto ch = exact_symbols.find(search);
  if (ch != TOKEN_NOT_FOUND) return std::string(1, ch);
  auto str = inexact_symbols.find(search);
  if (str.size()) return str;
  str = keywords.find(search);
  if (str.size()) return str;
  str = types.find(search);
  if (str.size()) return str;
  str = tags.find(search);
  if (str.size()) return str;
  return {};
}

int Tokens::findInAll(const std::string & search) const
{
  if (search.empty()) return TOKEN_NOT_FOUND;
  if (search.size()==1) {
    auto tok = exact_symbols.find(search[0]);
    if (tok != TOKEN_NOT_FOUND) return tok;
  }
  auto tok = inexact_symbols.find(search);
  if (tok != TOKEN_NOT_FOUND) return tok;
  tok = keywords.find(search);
  if (tok != TOKEN_NOT_FOUND) return tok;
  tok = types.find(search);
  if (tok != TOKEN_NOT_FOUND) return tok;
  tok = tags.find(search);
  if (tok != TOKEN_NOT_FOUND) return tok;
  return TOKEN_NOT_FOUND;
}


//==============================================================================
// Install the tokens
//==============================================================================
void Tokens::setup() {

  // add non-keywords here
  TokenMap = {
    { tok_eq, "==" },
    { tok_ne, "!=" },
    { tok_le, "<=" },
    { tok_ge, ">=" },
    { tok_asgmt_add, "+=" },
    { tok_asgmt_sub, "-=" },
    { tok_asgmt_mul, "*=" },
    { tok_asgmt_div, "/=" },
    { tok_eof, "eof" },
    { tok_ident, "identifier" },
    { tok_char_lit, "char_lit" },
    { tok_int_lit, "integer_lit" },
    { tok_real_lit, "real_lit" },
    { tok_string_lit, "string_lit" },
  };

  // add keywords here
  std::map<int, std::string> Keywords = {
    { tok_binary, "binary" },
    { tok_break, "break" },
    { tok_elif, "elif" },
    { tok_else, "else" },
    { tok_false, "false" },
    { tok_for, "for" },
    { tok_foreach, "foreach" },
    { tok_function, "fn" },
    { tok_if, "if" },
    { tok_reduce, "reduce" },
    { tok_return, "return" },
    { tok_task, "tsk" },
    { tok_true, "true" },
    { tok_unary, "unary" },
    { tok_use, "use" },
  };
    
  std::map<int, std::string> TypeKeywords = {
    { tok_i64, "i64" },
    { tok_f64, "f64" },
  };

  // create keyword list
  KeywordToToken.clear();
  for ( const auto & key_pair : Keywords )
    KeywordToToken.emplace( key_pair.second, key_pair.first );
  
  // create type keyword list
  TypeKeywordToToken.clear();
  for ( const auto & key_pair : TypeKeywords )
    TypeKeywordToToken.emplace( key_pair.second, key_pair.first );

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
  auto it = KeywordToToken.find(Name);
  if (it != KeywordToToken.end())
    return {true, it->second };
  
  return {false, 0};
}
//==============================================================================
// get a token from its name
//==============================================================================
bool Tokens::isType(int tok)
{ return TypeTokenToKeyword.count(tok); }


} // namespace
