#include "errors.hpp"
#include "lexer.hpp"
#include "string_utils.hpp"
#include "token.hpp"

#include <cstdio>
#include <iostream>

namespace contra {

//==============================================================================
// Get the next char
//==============================================================================
int Lexer::advance() {
  int LastChar_ = readchar();

  if (LastChar_ == '\n' || LastChar_ == '\r')
    LexLoc_.newLine();
  else
    LexLoc_.incrementCol();
  return LastChar_;
}


//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
int Lexer::gettok() {

  // Skip any whitespace.
  while (isspace(LastChar_))
    LastChar_ = advance();
  
  CurLoc_ = LexLoc_;

  //----------------------------------------------------------------------------
  // identifier: [a-zA-Z][a-zA-Z0-9]*
  if (isalpha(LastChar_)) {
    IdentifierStr_ = LastChar_;
    while (isalnum((LastChar_ = advance())))
      IdentifierStr_ += LastChar_;

    for ( int i=0; i<num_keywords; ++i )
      if (IdentifierStr_ == getTokName(tok_keywords[i]))
        return tok_keywords[i];
    return tok_identifier;
  }

  //----------------------------------------------------------------------------
  // Number: [0-9.]+
  if (isdigit(LastChar_) || LastChar_ == '.' || LastChar_ == '+' || LastChar_ == '-') {

    IdentifierStr_.clear();

    // peak if this is a unary or number
    if (LastChar_ == '+' || LastChar_ == '-') {
      auto NextChar = peek();
      if ( !isdigit(NextChar) && NextChar != '.' ) {
        int ThisChar = LastChar_;
        LastChar_ = advance();
        return ThisChar;
      }
    }

    // read first part of number
    bool is_float = (LastChar_ == '.');
    do {
      IdentifierStr_ += LastChar_;
      LastChar_ = advance();
      if (LastChar_ == '.') {
        if (is_float)
          THROW_SYNTAX_ERROR( "Multiple '.' encountered in real", LexLoc_.getLine() );
        is_float = true;
        // eat '.'
        IdentifierStr_ += LastChar_;
        LastChar_ = advance();
      }
    } while (isdigit(LastChar_));

    if (LastChar_ == 'e' || LastChar_ == 'E') {
      is_float = true;
      // eat e/E
      IdentifierStr_ += LastChar_;
      LastChar_ = advance();
      // make sure next character is sign or number
      if (LastChar_ != '+' && LastChar_ != '-' && !isdigit(LastChar_))
        THROW_SYNTAX_ERROR( "Digit or +/- must follow exponent", LexLoc_.getLine() );
      // eat sign or number
      IdentifierStr_ += LastChar_;
      LastChar_ = advance();
      // only numbers should follow
      do {
        IdentifierStr_ += LastChar_;
        LastChar_ = advance();
      } while (isdigit(LastChar_) );
    }

    if (is_float)
      return tok_real_number;
    else
      return tok_int_number;
  }

  //----------------------------------------------------------------------------
  // Comment until end of line.
  if (LastChar_ == '#') {
    do
      LastChar_ = advance();
    while (LastChar_ != eof() && LastChar_ != '\n' && LastChar_ != '\r');

    if (LastChar_ != eof())
      return gettok();
  }

  //----------------------------------------------------------------------------
  // string literal
  if (LastChar_ == '\"') {
    std::string quoted;
    while ((LastChar_ = advance()) != '\"')
      quoted += LastChar_;
    IdentifierStr_ = unescape(quoted);
    LastChar_ = advance();
    return tok_string;
  }

  //----------------------------------------------------------------------------
  // Check for end of file.  Don't eat the EOF.
  if (LastChar_ == eof())
    return tok_eof;

  //----------------------------------------------------------------------------
  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar_;
  LastChar_ = advance();
  return ThisChar;
}

} // namespace
