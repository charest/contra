#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include "sourceloc.hpp"
#include "token.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace contra {


//==============================================================================
/// Token position info
//==============================================================================
struct token_pos_t {
  std::ios::pos_type begin, end;
};

//==============================================================================
/// The lexer return datatype
//==============================================================================
struct lexer_results_t {
  std::vector<int> tokens;
  std::vector<token_pos_t> token_pos;

  std::string identifier_chars;
  std::vector<size_t> identifier_offsets;
  std::vector<int> identifier_to_token;

  size_t numTokens() const { return tokens.size(); }
  size_t numIdentifiers() const { return identifier_to_token.size(); }

  int findIdentifier(int tok) const;
  std::string getIdentifierString(int i) const;
};

/// Main lexer function
lexer_results_t lex(const Tokens & toks, std::istream& stream);
  
/// Dump lexer results
void print(std::ostream& os, const Tokens & toks, const lexer_results_t & res);

//==============================================================================
/// Return type for Lexer::gettok
//==============================================================================
struct token_info_t {
  int token;
  std::ios::pos_type begin, end;
  std::string identifier;
};

//==============================================================================
/// The lexer turns the text into tokens
//==============================================================================
class Lexer {

  /// The last character read
  int LastChar_ = ' ';

  std::istream *In_ = &std::cin;

  Tokens Tokens_; // TODO REF

  /// TODO Keep track of the location in the file
  SourceLocation LexLoc_;
  // TODO Where the identifier started (lags LexLoc)
  SourceLocation CurLoc_;

  std::stringstream Tee_;
  
  /// private helper function to get token and identifier
  int gettok(int & LastChar, std::string & IdentifierStr);

public:
  
  // constructor for reading from stdin
  Lexer() = default;

  // constructor from a stream
  Lexer( const Tokens & toks, std::istream & s ) : In_(&s), Tokens_(toks)
  {}

  /// read the next character
  char readchar() { return In_->get(); };
  std::string readline();
  char peek() { return In_->peek(); };
  bool eof() { return In_->eof(); }

  /// gettok - Return the next token from standard input.
  token_info_t gettok();

  // get the next character
  int advance();

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
