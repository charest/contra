#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include "stream.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace contra {

struct token_map_t;
  
//std::vector<std::string_view> identifiers;
//std::unordered_map<int, int> token_to_identifier;
//size_t numIdentifiers() const { return identifiers.size(); }

//int findIdentifier(int tok) const;
//std::string_view getIdentifierString(int i) const;

//==============================================================================
/// The lexer return datatype
//==============================================================================
struct lexed_t {
  std::vector<int> tokens;
  std::vector<stream_pos_t> token_pos;

  std::unordered_map<std::string, int> identifier_map;

  void add(int tok, stream_pos_t pos)
  {
    tokens.push_back( tok );
    token_pos.emplace_back( pos );
  }

  size_t size() const { return tokens.size(); }
};

/// Main lexer function
int lex(const stream_t & stream, lexed_t & lx);
void recognize(const stream_t & stream, const token_map_t & tmap, lexed_t & lx);
  
/// Dump lexer results
void print(std::ostream& os, const stream_t & stream, const lexed_t & res);

} // namespace

#endif // CONTRA_LEXER_HPP
