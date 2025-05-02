#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace contra {

struct stream_t;
struct stream_pos_t;
struct token_map_t;

//==============================================================================
/// The lexer return datatype
//==============================================================================
struct lexed_t {
  std::vector<int> tokens;
  std::vector<stream_pos_t> token_pos;

  std::unordered_map<std::string, int> identifier_map;
  std::vector<std::string_view> identifiers;
  std::unordered_map<int, int> token_to_identifier;

  void add(int tok, stream_pos_t pos, const std::string & str = "");

  size_t numTokens() const { return tokens.size(); }
  size_t numIdentifiers() const { return identifiers.size(); }

  int findIdentifier(int tok) const;
  std::string_view getIdentifierString(int i) const;
};

/// Main lexer function
int lex(stream_t & stream, const token_map_t & toks, lexed_t & lx);
  
/// Dump lexer results
void print(std::ostream& os, const lexed_t & res);

} // namespace

#endif // CONTRA_LEXER_HPP
