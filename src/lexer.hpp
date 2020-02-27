#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include "sourceloc.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace contra {

class Lexer {

  int LastChar = ' ';
  SourceLocation LexLoc = {1, 0};

  std::ifstream InputStream;
  std::istream *In = &std::cin;

public:
  
  SourceLocation CurLoc;
  std::string IdentifierStr; // Filled in if tok_identifier

  Lexer() = default;
  Lexer( const std::string & filename ){
    InputStream.open(filename.c_str());
    if (!InputStream.good()) {
      std::stringstream ss;
      ss << "File '" << filename << "' does not exists" << std::endl;
      throw std::runtime_error( ss.str() );
    }
    In = &InputStream; 
  }

  ~Lexer() { if (InputStream) InputStream.close(); }

  /// read the next character
  int readchar() { return In->get(); };
  int peek() { return In->peek(); };
  auto eof() { return In->eof(); }

  /// gettok - Return the next token from standard input.
  int gettok();

  // get the next character
  int advance();

};

} // namespace

#endif // CONTRA_LEXER_HPP
