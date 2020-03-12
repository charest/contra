#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include "sourceloc.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace contra {

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

public:

  // constructor for reading from stdin
  Lexer() = default;

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
  int gettok();

  // get the next character
  int advance();

  // get the source location
  const SourceLocation & getLexLoc() const { return LexLoc_; }
  // get the current location
  const SourceLocation & getCurLoc() const { return CurLoc_; }

  // get the identifier string
  const std::string & getIdentifierStr() const
  { return IdentifierStr_; }

  // print out current line
  std::ostream & barf(std::ostream& out, SourceLocation Loc);
};

} // namespace

#endif // CONTRA_LEXER_HPP
