#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include "sourceloc.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace contra {

//==============================================================================
/// The lexer return datatype
//==============================================================================
struct token_pos_t {
  std::ios::pos_type begin, end;
};

struct lexer_results_t {
  std::vector<int> tokens;
  std::vector<token_pos_t> token_pos;

  std::string identifier_chars;
  std::vector<size_t> identifier_offsets;
  std::vector<int> identifier_to_token;
};

lexer_results_t lex(std::istream& stream);

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
  /// Keep track of the location in the file
  SourceLocation LexLoc_;

  std::ifstream InputStream_;
  std::istream *In_ = &std::cin;

  // Where the identifier started (lags LexLoc)
  SourceLocation CurLoc_;
  // Filled in if tok_identifier
  std::string IdentifierStr_;

  std::stringstream Tee_;
  std::string FileName_ = "<stdin>";
  
  /// private helper function to get token and identifier
  int gettok(std::string & IdentifierStr);

public:

  // constructor for reading from stdin
  Lexer() = default;
  
  // constructor from a stream
  Lexer( std::istream & s ) : In_(&s)
  {}

  // constructor cor reading from file
  Lexer( const std::string & filename ) : FileName_(filename)
  {
    InputStream_.open(filename.c_str());
    if (!InputStream_.good()) {
      std::stringstream ss;
      ss << "File '" << filename << "' does not exists" << std::endl;
      throw std::runtime_error( ss.str() );
    }
    In_ = &InputStream_; 
  }

  ~Lexer() { if (InputStream_) InputStream_.close(); }

  /// read the next character
  int readchar() { return In_->get(); };
  std::string readline();
  int peek() { return In_->peek(); };
  int eof() { return In_->eof(); }

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


  // get the identifier string
  const std::string & getIdentifierStr() const
  { return IdentifierStr_; }

  // print out current line
  std::ostream & barf(std::ostream& out, const LocationRange & Loc);
};

} // namespace

#endif // CONTRA_LEXER_HPP
