#include "lexer.hpp"
#include "token.hpp"

#include <cstdio>
#include <iostream>

namespace contra {


//==============================================================================
// Get the next char
//==============================================================================
int Lexer::advance() {
  int LastChar = readchar();

  if (LastChar == '\n' || LastChar == '\r') {
    LexLoc.Line++;
    LexLoc.Col = 0;
  }
  else {
    LexLoc.Col++;
  }
  return LastChar;
}


//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
int Lexer::gettok() {

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = advance();
  
  CurLoc = LexLoc;

  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    while (isalnum((LastChar = advance())))
      IdentifierStr += LastChar;

    for ( int i=0; i<num_keywords; ++i )
      if (IdentifierStr == getTokName(tok_keywords[i]))
        return tok_keywords[i];
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = advance();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
      LastChar = advance();
    while (LastChar != eof() && LastChar != '\n' && LastChar != '\r');

    if (LastChar != eof())
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == eof())
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = advance();
  return ThisChar;
}

} // namespace
