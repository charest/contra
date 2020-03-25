#include "errors.hpp"
#include "lexer.hpp"
#include "string_utils.hpp"
#include "token.hpp"

#include <cstdio>
#include <iostream>

namespace contra {
    
//==============================================================================
// Read the rest of the line
//==============================================================================
std::string Lexer::readline()
{
  std::string tmp;
  std::getline(*In_, tmp);
  return tmp;
}

//==============================================================================
// Get the next char
//==============================================================================
int Lexer::advance() {
  int LastChar = readchar();
  Tee_ << static_cast<char>(LastChar);

  if (LastChar == '\n' || LastChar == '\r')
    LexLoc_.newLine();
  else
    LexLoc_.incrementCol();
  return LastChar;
}


//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
int Lexer::gettok() {

  // Skip any whitespace.
  while (isspace(LastChar_))
    LastChar_ = advance();
  
  auto NextChar = peek();
  CurLoc_ = LexLoc_;

  //----------------------------------------------------------------------------
  // identifier: [a-zA-Z][a-zA-Z0-9]*
  if (isalpha(LastChar_)) {
    IdentifierStr_ = LastChar_;
    while (isalnum((LastChar_ = advance())) || LastChar_=='_')
      IdentifierStr_ += LastChar_;

    auto res = Tokens::getTok(IdentifierStr_);
    if (res.found) return res.token;
    
    return tok_identifier;
  }

  //----------------------------------------------------------------------------
  // Number: [0-9.]+
  //if (isdigit(LastChar_) || LastChar_ == '.') { // Number: [0-9.]+
  //  std::string NumStr;
  //  do {
  //    NumStr += LastChar_;
  //    LastChar_ = advance();
  //  } while (isdigit(LastChar_) || LastChar_ == '.');

  //  IdentifierStr_ = NumStr; 
  //  return tok_int_number;
  //}

  // check if there is a sign in from of a number
  //bool is_signed_number = false;
  //if (LastChar_ == '+' || LastChar_ == '-')
  //  is_signed_number = isdigit(NextChar) || NextChar == '.';
    
  if (isdigit(LastChar_) || LastChar_ == '.' /*|| is_signed_number*/) {

    IdentifierStr_.clear();

    // eat the sign if it has one
    //if (is_signed_number) {
    //  IdentifierStr_ += LastChar_;    
    //  LastChar_ = advance();
    //}

    // read first part of number
    bool is_float = (LastChar_ == '.');
    do {
      IdentifierStr_ += LastChar_;
      LastChar_ = advance();
      if (LastChar_ == '.') {
        if (is_float)
          THROW_SYNTAX_ERROR( "Multiple '.' encountered in real", LexLoc_ );
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
        THROW_SYNTAX_ERROR( "Digit or +/- must follow exponent", LexLoc_ );
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
  // Comparison operators
  if (LastChar_ == '=' && NextChar == '=') {
    advance(); // eat next =
    LastChar_ = advance();
    return tok_eq;
  }
  if (LastChar_ == '<' && NextChar == '=') {
    advance(); // eat next =
    LastChar_ = advance();
    return tok_le;
  }
  if (LastChar_ == '>' && NextChar == '=') {
    advance(); // eat next =
    LastChar_ = advance();
    return tok_ge;
  }
  if (LastChar_ == '!' && NextChar == '=') {
    advance(); // eat next =
    LastChar_ = advance();
    return tok_ne;
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

//==============================================================================
/// dump out the current line
//==============================================================================
std::ostream & Lexer::barf(std::ostream& out, SourceLocation Loc)
{
  auto max = std::numeric_limits<std::streamsize>::max();
  // finish the line
  Tee_ << readline(); 
  // skip lines
  auto line = Loc.getLine(); 
  auto col = Loc.getCol();
  auto prev_col = std::max(col - 1, 0);
  for ( int i=0; i<line-1; ++i ) Tee_.ignore(max, '\n');
  // get relevant line
  std::string tmp;
  std::getline(Tee_, tmp);
  // start output
  out << FileName_ << " : Line " << line << " : Col " << col << ":" << std::endl;
  out << tmp << std::endl;
  out << std::string(prev_col, ' ') << "^" << std::endl;
  return out;
}


} // namespace
