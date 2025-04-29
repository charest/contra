#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include "sourceloc.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace contra {

struct Tokens;
struct stream_t;
struct stream_pos_t;

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
int lex(const Tokens & toks, stream_t & stream, lexed_t & lx);
  
/// Dump lexer results
void print(std::ostream& os, const Tokens & toks, const lexed_t & res);


//==============================================================================
/// The lexer turns the text into tokens
//==============================================================================
class Lexer {

  /// The last character read
  int LastChar_ = ' ';

  std::istream *In_ = &std::cin;

  /// private helper function to get token and identifier
  int gettok(int & LastChar, std::string & IdentifierStr) { return 0; }

public:
  
  // constructor for reading from stdin
  Lexer() = default;

  // constructor from a stream
  Lexer( const Tokens & toks, std::istream & s ) : In_(&s)
  {}

  /// read the next character
  char advance() { return In_->get(); };
  std::string readline() { return ""; }
  char peek() { return In_->peek(); };
  bool eof() { return In_->eof(); }

  // TODO DELETE ALL THIS
  /// TODO Keep track of the location in the file
  SourceLocation LexLoc_;
  // TODO Where the identifier started (lags LexLoc)
  SourceLocation CurLoc_;

  std::stringstream Tee_;
  
  // get the source location
  const SourceLocation & getLexLoc() const { return LexLoc_; }
  // get the current location
  const SourceLocation & getCurLoc() const { return CurLoc_; }
  // get both locations as a range
  LocationRange getIdentifierLoc() const
  { return LocationRange(CurLoc_, LexLoc_); }


  // print out current line
  std::ostream & barf(std::ostream& out, const LocationRange & Loc);
};

} // namespace

#endif // CONTRA_LEXER_HPP
