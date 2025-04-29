#include "errors.hpp"
#include "lexer.hpp"
#include "stream.hpp"
#include "token.hpp"

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
void print(std::ostream& os, const Tokens & toks, const lexed_t & res)
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
    auto tystr = toks.findInAll(tyid);
    
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
/// gettok - Return the next token from standard input.
//==============================================================================
int gettok(
  const Tokens & toks,
  stream_t & is,
  int & LastChar,
  int & tok,
  std::string & IdentifierStr)
{

  auto & in = is.in;
  auto NextChar = in.peek();
  IdentifierStr.clear();
  int err = 0;

  //----------------------------------------------------------------------------
  // identifier: [a-zA-Z][a-zA-Z0-9]*
  if (isalpha(LastChar)) {

    std::string str(1, LastChar);
    while (isalnum((LastChar = advance(in))) || LastChar=='_')
      str += LastChar;

    tok = toks.keywords.find(str);
    if (tok!=TOKEN_NOT_FOUND) return err;
    
    tok = toks.types.find(str);
    if (tok!=TOKEN_NOT_FOUND) return err;
    
    IdentifierStr = str;
    tok = toks.identifier;
    return err;
  }
  
  //----------------------------------------------------------------------------
  // Number: [0-9.]+

  // check if there is a sign in from of a number
  //bool is_signed_number = false;
  //if (LastChar == '+' || LastChar == '-')
  //  is_signed_number = isdigit(NextChar) || NextChar == '.';
    
  if (isdigit(LastChar) || LastChar == '.' /*|| is_signed_number*/) {

    // eat the sign if it has one
    //if (is_signed_number) {
    //  IdentifierStr += LastChar;    
    //  LastChar = advance(in);
    //}

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
      if (LastChar != '+' && LastChar != '-' && !isdigit(LastChar)) {
        err += error( is, "Digit or +/- must follow exponent" );
      }
      else {
        // eat sign or number
        IdentifierStr += LastChar;
        LastChar = advance(in);
        // only numbers should follow
        do {
          IdentifierStr += LastChar;
          LastChar = advance(in);
        } while (isdigit(LastChar) );
      }
    }
    tok = is_float ? toks.real_literal : toks.int_literal;
    return err;
  }

  //----------------------------------------------------------------------------
  // Comment until end of line.
  if (LastChar == toks.comment) {
    do
      LastChar = advance(in);
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    tok = toks.comment;
    return err;
  }

  //----------------------------------------------------------------------------
  // string literal
  if (LastChar == toks.quote) {
    std::string quoted;
    while ((LastChar = advance(in)) != toks.quote)
      quoted += LastChar;
    IdentifierStr = utils::unescape(quoted);
    LastChar = advance(in);
    tok = toks.string_literal;
    return err;
  }
  
  //----------------------------------------------------------------------------
  // Operators

  // two character operators
  auto char_as_str = std::string(1,LastChar);
  {
    auto str = char_as_str + static_cast<char>(NextChar);
    tok = toks.inexact_symbols.find(str);

    if (tok != TOKEN_NOT_FOUND) {
      advance(in); // eat next =
      LastChar = advance(in);
      return err;
    }
  } 
  
  // single character operators
  {
    tok = toks.inexact_symbols.find(char_as_str);

    if (tok != TOKEN_NOT_FOUND) {
      LastChar = advance(in);
      return err;
    }
  }

  //----------------------------------------------------------------------------
  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF) {
    tok = toks.eof;
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
  const Tokens & toks,
  stream_t & is,
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
  auto err = gettok(toks, is, last_char, tok, identifier);
  pos.end = in.tellg();
  
  return err;
}

//==============================================================================
// Main function to generate tokens from a stream
//==============================================================================
int lex(const Tokens & toks, stream_t & in, lexed_t & res)
{
  int err = 0;
  std::string identifier;
  int tok;
  int last_char = ' ';
  stream_pos_t pos;
  do
  {
    err += gettok(toks, in, last_char, tok, pos, identifier);
    res.add( tok, pos, identifier );
  } while (tok!=toks.eof);
    
  return err;
}


//==============================================================================
/// dump out the current line
//==============================================================================
std::ostream & Lexer::barf(std::ostream& out, const LocationRange & Loc)
{
  auto max = std::numeric_limits<std::streamsize>::max();
  // finish the line
  Tee_ << readline(); 
  // check begin and end
  const auto & BegLoc = Loc.getBegin();
  const auto & EndLoc = Loc.getEnd();
  // skip lines
  auto BegLine = BegLoc.getLine(); 
  auto BegCol = BegLoc.getCol();
  auto PrevCol = std::max(BegCol - 1, 0);
  for ( int i=0; i<BegLine-1; ++i ) Tee_.ignore(max, '\n');
  // get relevant line
  std::string tmp;
  std::getline(Tee_, tmp);
  // start output
  out << "Line " << BegLine << " : Col " << BegCol << ":" << std::endl;
  out << tmp << std::endl;
  out << std::string(PrevCol, ' ') << "^";
  if (BegLine == EndLoc.getLine()) {
    auto EndCol = EndLoc.getCol();
    auto Len = std::max(EndCol-1 - PrevCol-1, 0);
    out << std::string(Len-1, '~');
  }
  out << std::endl;
  return out;
}


} // namespace
