#include "errors.hpp"
#include "lexer.hpp"
#include "stream.hpp"
#include "token.hpp"
#include "toks.hpp"

#include "utils/string_utils.hpp"

#include <cstdio>
#include <iostream>
#include <iomanip>

namespace contra {
  
/// Get an identifier string
std::string_view lexed_t::getIdentifierString(int i) const
{ 
  if (i<0 || i >= identifiers.size()) return {};
  return identifiers[i];
}

/// Find an identifier from a token
int lexed_t::findIdentifier(int tok) const
{
  auto it = token_to_identifier.find(tok);
  if (it != token_to_identifier.end()) return it->second;
  return -1;
}

/// Add the identifier string
void lexed_t::add(int token, stream_pos_t pos, const std::string & identifier)
{
    if (identifier.size()) {
      auto nidents = identifiers.size();
      auto ntoks = tokens.size();
      // try to insert the identifier
      auto res = identifier_map.try_emplace( identifier, nidents );
      // if new, add it to the vector as well
      if (res.second) identifiers.emplace_back( res.first->first );
      // add the token mapping
      token_to_identifier[ntoks] = res.first->second;
    }
    tokens.push_back( token );
    token_pos.emplace_back( pos );
}


//==============================================================================
// Lexer output operator
//==============================================================================
void print(std::ostream& os, const lexed_t & res)
{
  using utils::printRight, utils::printLeft;
  auto n = res.tokens.size();
  int digits = utils::count_digits(n);
  auto aw = std::max(digits+1, 7);
  auto bw = 6;
  auto cw = 14;
  auto dw = std::max(bw, 8);
  auto ew = 4*aw;

  printRight(os, aw, ' ', "TokenId");
  printRight(os, 2, ' ');
  printRight(os, bw, ' ', "TypeId");
  printRight(os, 2, ' ');
  printRight(os, cw, ' ', "TypeString");
  printRight(os, 2, ' ');
  printRight(os, dw, ' ', "IndentId");
  printRight(os, 2, ' ');
  printLeft (os, ew, ' ', "IdentString");
  os << std::endl;

  printRight(os, aw, '-');
  printRight(os, 2, ' ');
  printRight(os, bw, '-');
  printRight(os, 2, ' ');
  printRight(os, cw, '-');
  printRight(os, 2, ' ');
  printRight(os, dw, '-');
  printRight(os, 2, ' ');
  printLeft (os, ew, '-');
  os << std::endl;

  for (size_t i=0; i<n; ++i) {
    auto id = res.findIdentifier(i);
    auto tyid = res.tokens[i];
    auto tystr = tok_to_string(tyid);

    std::stringstream ss;
    if (id>=0)
      ss << "\"" << res.getIdentifierString(id) << "\"";

    printRight(os, aw, ' ', i);
    printRight(os, 2, ' ');
    printRight(os, bw, ' ', tyid);
    printRight(os, 2, ' ');
    printRight(os, cw, ' ', tystr);
    printRight(os, 2, ' ');
    if (id>=0)
      printRight(os, dw, ' ', id);
    else
      printRight(os, dw, ' ');
    printRight(os, 2, ' ');
    printLeft (os, ew, ' ', ss.str());
    os << std::endl;
  }

}
 
//==============================================================================
/// Advance the file position
//==============================================================================
int advance(std::istream & in)
{ return in.get(); }
    

//==============================================================================
/// Read until a specified position
//==============================================================================
int get_until(std::istream & in, int c, std::string & str)
{
  int LastChar;
  while ((LastChar = advance(in)) != c)
    str += LastChar;
  return LastChar;
}
    
int a_or_ab(
  std::istream & in,
  int LastChar,
  int NextChar,
  int NextSym,
  int NextLabel,
  int & tok)
{
  if (NextChar == NextSym) {
    tok = NextLabel;
    advance(in);
  }
  else
    tok = LastChar;
  return advance(in);
}
  
//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
int gettok(
  stream_t & is,
  const token_map_t & toks,
  int & LastChar,
  int & tok,
  std::string & IdentifierStr)
{

  auto & in = is.in;
  auto NextChar = in.peek();
  int err = 0;
  IdentifierStr.clear();
  tok = TOK_UNK;
  
  //----------------------------------------------------------------------------
  // identifier: [a-zA-Z][a-zA-Z0-9]*
  if (std::isalpha(LastChar)) {

    IdentifierStr += LastChar;
    while (std::isalnum((LastChar = advance(in))) || LastChar=='_')
      IdentifierStr += LastChar;

    tok = toks.find(IdentifierStr);
    if (tok!=TOKEN_NOT_FOUND) {
      IdentifierStr.clear();
      return err;
    }

    tok = TOK_IDENT;
    return err;
  }
  
  //----------------------------------------------------------------------------
  // Number: [0-9.]+

  if (std::isdigit(LastChar) || (LastChar == '.' && std::isdigit(NextChar))) {

    // read first part of number
    int numDec = (LastChar == '.');
    do {
      IdentifierStr += LastChar;
      LastChar = advance(in);
      auto has_dec = (LastChar == '.');
      if (numDec == 1 && has_dec)
        err += error( is, "Multiple '.' encountered in real" );
      numDec += has_dec;
    } while (std::isdigit(LastChar) || LastChar == '.');

    bool is_float = numDec;

    if (LastChar == 'e' || LastChar == 'E') {
      is_float = true;
      // eat e/E
      IdentifierStr += LastChar;
      LastChar = advance(in);
      // make sure next character is sign or number
      auto isSign = (LastChar == '+') || (LastChar == '-');
      if (!isSign && !std::isdigit(LastChar))
        err += error( is, "Digit or +/- must follow exponent" );
      // eat sign or number
      IdentifierStr += LastChar;
      LastChar = advance(in);
      // if it was a sign, there has to be a number
      if (isSign && !std::isdigit(LastChar))
        err += error( is, "Digit must follow exponent sign" );
      // only numbers should follow
      while (std::isdigit(LastChar)) {
        IdentifierStr += LastChar;
        LastChar = advance(in);
      }
    }
    tok = is_float ? TOK_REAL_LIT : TOK_INT_LIT;
    return err;
  }

  switch (LastChar) {

  //----------------------------------------------------------------------------
  // Comment until end of line.
  case '#':
  
    do
      LastChar = advance(in);
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    tok = TOK_COMMENT;
    return err;
  
  
  //----------------------------------------------------------------------------
  // string literal
  case '\"':

    LastChar = get_until(in, '\"', IdentifierStr);
    LastChar = advance(in);
    tok = TOK_STRING_LIT;
    return err;
  
  //----------------------------------------------------------------------------
  // Operators

  case '+':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_ADD_EQ, tok);
    return err;
  
  case '-':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_SUB_EQ, tok);
    return err;
  
  case '*':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_MUL_EQ, tok);
    return err;
  
  case '/':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_DIV_EQ, tok);
    return err;
  
  case '=':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_EQUIV, tok);
    return err;
  
  case '!':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_NE, tok);
    return err;
  
  case '<':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_LE, tok);
    return err;
  
  case '>':
    LastChar = a_or_ab(in, LastChar, NextChar, '=', TOK_GE, tok);
    return err;
  
  //----------------------------------------------------------------------------
  // Check for end of file.  Don't eat the EOF.
  case EOF:

    tok = TOK_EOF;
    return err;
  
  }

  //----------------------------------------------------------------------------
  // Otherwise, just return the character as its ascii value.
  tok = LastChar;
  LastChar = advance(in);
  return err;
}

//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
int gettok(
  stream_t & is,
  const token_map_t & toks,
  int & last_char,
  int & tok,
  stream_pos_t & pos,
  std::string & identifier)
{
  auto & in = is.in;

  // Skip any whitespace.
  while (isspace(last_char))
    last_char = advance(in);

  pos.begin = in.tellg();
  auto err = gettok(is, toks, last_char, tok, identifier);
  pos.end = in.tellg();
  
  return err;
}

//==============================================================================
// Main function to generate tokens from a stream
//==============================================================================
int lex(stream_t & in, const token_map_t & toks, lexed_t & res)
{
  int err = 0;
  std::string identifier;
  int tok;
  int last_char = ' ';
  stream_pos_t pos;
  do
  {
    err += gettok(in, toks, last_char, tok, pos, identifier);
    res.add( tok, pos, identifier );
  } while (tok!=TOK_EOF);
    
  return err;
}

} // namespace
