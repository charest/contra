#ifndef CONTRA_LEXER_HPP
#define CONTRA_LEXER_HPP

#include <string>

namespace contra {

class Lexer {

  int LastChar = ' ';

public:
  
  std::string IdentifierStr; // Filled in if tok_identifier
  double NumVal;             // Filled in if tok_number

  /// gettok - Return the next token from standard input.
  int gettok();

};

} // namespace

#endif // CONTRA_LEXER_HPP
