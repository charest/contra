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

//==============================================================================
// Lexer output operator
//==============================================================================
void print(std::ostream& os, const stream_t & stream, const lexed_t & res)
{
  using utils::printRight, utils::printLeft;
  auto n = res.tokens.size();
  int digits = utils::count_digits(n);
  auto aw = std::max(digits+1, 7);
  auto bw = 6;
  auto cw = 14;
  auto ew = 4*aw;

  printRight(os, aw, ' ', "TokenId");
  printRight(os, 2, ' ');
  printRight(os, bw, ' ', "TypeId");
  printRight(os, 2, ' ');
  printRight(os, cw, ' ', "TypeString");
  printRight(os, 2, ' ');
  printLeft (os, ew, ' ', "IdentString");
  os << std::endl;

  printRight(os, aw, '-');
  printRight(os, 2, ' ');
  printRight(os, bw, '-');
  printRight(os, 2, ' ');
  printRight(os, cw, '-');
  printRight(os, 2, ' ');
  printRight(os, ew, '-');
  os << std::endl;

  for (size_t i=0; i<n; ++i) {
    auto tyid = res.tokens[i];
    auto pos = res.token_pos[i];
    auto tystr = tok_to_string(tyid);

    std::stringstream ss;
    switch (tyid) {
    case (TOK_IDENT):
    case (TOK_INT_LIT):
    case (TOK_REAL_LIT):
    case (TOK_STRING_LIT):
      ss << "\"" << stream.at(pos) << "\"";
    }

    printRight(os, aw, ' ', i);
    printRight(os, 2, ' ');
    printRight(os, bw, ' ', tyid);
    printRight(os, 2, ' ');
    printRight(os, cw, ' ', tystr);
    printRight(os, 2, ' ');
    printLeft (os, ew, ' ', ss.str());
    os << std::endl;
  }

}
 
std::tuple<int,size_t,int>
a_or_ab(
  const std::string & buffer,
  size_t cur,
  int NextSym,
  int NextLabel,
  int err)
{
  auto tok = buffer[cur];
  auto LastChar = buffer[++cur];
  if (LastChar == NextSym)
    return {NextLabel, ++cur, err};
  else
    return {tok, cur, err};
}
  
//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
std::tuple<int,size_t,int>
gettok( const stream_t & is, size_t cur )
{
  auto & buffer = is.buffer;
  auto LastChar = buffer[cur];
  int err = 0;
  
  //----------------------------------------------------------------------------
  // identifier: [a-zA-Z][a-zA-Z0-9]*
  if (std::isalpha(LastChar)) {
     
    do {
      LastChar = buffer[++cur];
    } while (std::isalnum(LastChar) || LastChar=='_');

    return {TOK_IDENT, cur, err};
  }
  
  //----------------------------------------------------------------------------
  // Number: [0-9.]+

  if (std::isdigit(LastChar) || (LastChar == '.' && std::isdigit(buffer[cur+1]))) {

    // read first part of number
    int numDec = (LastChar == '.');
    do {
      LastChar = buffer[++cur];
      auto has_dec = (LastChar == '.');
      if (numDec == 1 && has_dec)
        err += error( is, "Multiple '.' encountered in real", cur );
      numDec += has_dec;
    } while (std::isdigit(LastChar) || LastChar == '.');

    bool is_float = numDec;

    if (LastChar == 'e' || LastChar == 'E') {
      is_float = true;
      // eat e/E
      LastChar = buffer[++cur];
      // make sure next character is sign or number
      auto isSign = (LastChar == '+') || (LastChar == '-');
      if (!isSign && !std::isdigit(LastChar))
        err += error( is, "Digit or +/- must follow exponent", cur );
      // eat sign or number
      LastChar = buffer[++cur];
      // if it was a sign, there has to be a number
      if (isSign && !std::isdigit(LastChar))
        err += error( is, "Digit must follow exponent sign", cur );
      // only numbers should follow
      while (std::isdigit(LastChar)) {
        LastChar = buffer[++cur];
      }
    }
    auto tok = is_float ? TOK_REAL_LIT : TOK_INT_LIT;
    return {tok, cur, err};
  }

  switch (LastChar) {

  //----------------------------------------------------------------------------
  // Comment until end of line.
  case '#':
  
    do {
      LastChar = buffer[++cur];
    } while (LastChar != '\0' && LastChar != '\n' && LastChar != '\r');

    return {TOK_COMMENT, cur, err};
  
  
  //----------------------------------------------------------------------------
  // string literal
  case '\"':
      
    LastChar = buffer[++cur];

    while (LastChar != '\"')
      LastChar = buffer[++cur];

    return {TOK_STRING_LIT, ++cur, err};
  
  //----------------------------------------------------------------------------
  // Operators

  case '+': return a_or_ab(buffer, cur, '=', TOK_ADD_EQ, err);
  case '-': return a_or_ab(buffer, cur, '=', TOK_SUB_EQ, err);
  case '*': return a_or_ab(buffer, cur, '=', TOK_MUL_EQ, err);
  case '/': return a_or_ab(buffer, cur, '=', TOK_DIV_EQ, err);
  case '=': return a_or_ab(buffer, cur, '=', TOK_EQUIV, err);
  case '!': return a_or_ab(buffer, cur, '=', TOK_NE, err);
  case '<': return a_or_ab(buffer, cur, '=', TOK_LE, err);
  case '>': return a_or_ab(buffer, cur, '=', TOK_GE, err);
  
  }

  //----------------------------------------------------------------------------
  // Otherwise, just return the character as its ascii value.
  return {LastChar, ++cur, err};
}

//==============================================================================
// Main function to generate tokens from a stream
//==============================================================================
int lex(const stream_t & in, lexed_t & lx)
{
  int err = 0;
  size_t cur = 0;
  auto & buffer = in.buffer;
  auto bufsize = in.buffer.size();
  stream_pos_t pos{0, 0};
    
  
  while (cur < bufsize)
  {
    // Skip any whitespace.
    while (isspace(buffer[cur])) cur++;

    if (cur >= bufsize) break;

    // get the next token
    pos.begin = cur;
    int e, tok;
    std::tie(tok, cur, e) = gettok(in, cur);
    err += e;
    pos.end = cur;

    switch (tok) {
    #define TOKS_CASE(name, str, ...) case name:
    FOR_LEX_STATES(TOKS_CASE)
    #undef TOKS_CASE
    
    case 0 ... 255:
      lx.add(tok, pos);
      break;
    }
  }

  lx.add(TOK_EOF, {pos.end, pos.end+1});
    
  return err;
}


//==============================================================================
// Remap identifiers
//==============================================================================
void recognize(const stream_t & stream, const token_map_t & toks, lexed_t & lx)
{
  auto & token_pos = lx.token_pos;
  auto & tokens = lx.tokens;
  auto ntok = tokens.size();
  const auto & tmap = toks.str_to_enum;
  const auto & buffer = stream.buffer;

  for (size_t i=0; i<ntok; ++i) {
    const auto & pos = token_pos[i];
    auto len = pos.length();
    if (tokens[i] == TOK_IDENT) {
      auto it = tmap.find( buffer.substr(pos.begin, len) );
      if (it != tmap.end()) tokens[i] = it->second;
    }
  }
}

} // namespace
