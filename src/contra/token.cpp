#include "token.hpp"

namespace contra {

// Initializers
Tokens::map_type Tokens::TokenMap = {};
Tokens::reverse_map_type Tokens::KeywordToToken = {};
Tokens::reverse_map_type Tokens::TypeKeywordToToken = {};
Tokens::map_type Tokens::TypeTokenToKeyword = {};
  

std::string Tokens::findInAll(int search) const
{
  auto str = one_char.find(search);
  if (str.size()) return str;
  str = multi_char.find(search);
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
  auto tok = one_char.find(search);
  if (tok != tok_not_found) return tok;
  tok = multi_char.find(search);
  if (tok != tok_not_found) return tok;
  tok = keywords.find(search);
  if (tok != tok_not_found) return tok;
  tok = types.find(search);
  if (tok != tok_not_found) return tok;
  tok = tags.find(search);
  if (tok != tok_not_found) return tok;
  return tok_not_found;
}


//==============================================================================
// Make contra tokens
//==============================================================================
Tokens make_contra_tokens() {
  Tokens toks;

  toks.one_char.add( tok_comment );
  toks.one_char.add( tok_sep );
  toks.one_char.add( tok_comma );
  toks.one_char.add( tok_colon );
  toks.one_char.add( tok_asgmt );
  toks.one_char.add( tok_lt );
  toks.one_char.add( tok_gt );
  toks.one_char.add( tok_add );
  toks.one_char.add( tok_sub );
  toks.one_char.add( tok_mul );
  toks.one_char.add( tok_div );
  toks.one_char.add( tok_mod );
  toks.one_char.add( tok_lparens );
  toks.one_char.add( tok_rparens );
  toks.one_char.add( tok_lbrack );
  toks.one_char.add( tok_rbrack );
    
  toks.multi_char.add( tok_eq, "==" );
  toks.multi_char.add( tok_ne, "!=" );
  toks.multi_char.add( tok_le, "<=" );
  toks.multi_char.add( tok_ge, ">=" );
  toks.multi_char.add( tok_asgmt_add, "+=" );
  toks.multi_char.add( tok_asgmt_sub, "-=" );
  toks.multi_char.add( tok_asgmt_mul, "*=" );
  toks.multi_char.add( tok_asgmt_div, "/=" );
  
  toks.keywords.add( tok_if, "if" );
  toks.keywords.add( tok_elif, "elif" );
  toks.keywords.add( tok_else, "else" );
  toks.keywords.add( tok_for, "for" );
  toks.keywords.add( tok_foreach, "foreach" );
  toks.keywords.add( tok_break, "break" );
  toks.keywords.add( tok_reduce, "reduce" );
  toks.keywords.add( tok_use, "use" );
  toks.keywords.add( tok_true, "true" );
  toks.keywords.add( tok_false, "false" );
  toks.keywords.add( tok_function, "fn" );
  toks.keywords.add( tok_return, "return" );
  toks.keywords.add( tok_task, "tsk" );
  
  toks.types.add( tok_i64, "i64" );
  toks.types.add( tok_f64, "f64" );
  
  toks.tags.add( tok_eof, "eof" );
  toks.tags.add( tok_identifier, "identifier" );
  toks.tags.add( tok_char_literal, "char_literal" );
  toks.tags.add( tok_int_literal, "integer_literal" );
  toks.tags.add( tok_real_literal, "real_literal" );
  toks.tags.add( tok_string_literal, "string_literal" ); 
    
  return toks;
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
    { tok_identifier, "identifier" },
    { tok_char_literal, "char_literal" },
    { tok_int_literal, "integer_literal" },
    { tok_real_literal, "real_literal" },
    { tok_string_literal, "string_literal" },
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
