#include "errors.hpp"
#include "lexer.hpp"
#include "token.hpp"

#include "utils/string_utils.hpp"

#include <cstdio>
#include <iostream>
#include <iomanip>

namespace contra {
  
/// Get an identifier string
std::string lexer_results_t::getIdentifierString(int i) const
{ 
  if (i<0) return {};
  auto start = identifier_offsets[i];
  auto end = identifier_offsets[i+1];
  return identifier_chars.substr(start, end-start);
}

/// Find an identifier from a token
int lexer_results_t::findIdentifier(int tok) const
{
  auto it = std::lower_bound(
    identifier_to_token.begin(),
    identifier_to_token.end(),
    tok);
  if (it!=identifier_to_token.end() && *it==tok)
    return std::distance(identifier_to_token.begin(), it);
  return -1;
}

//==============================================================================
// Main function to generate tokens from a stream
//==============================================================================
lexer_results_t lex(const Tokens & tok, std::istream & in)
{
  Lexer TheLex(tok, in); 

  lexer_results_t res;
  res.identifier_offsets.push_back(0);

  token_info_t ti;
  do
  {
    ti = TheLex.gettok();
    if (ti.identifier.size()) {
      res.identifier_chars += ti.identifier;
      res.identifier_offsets.push_back( res.identifier_chars.size() );
      res.identifier_to_token.emplace_back( res.tokens.size() );
    }
    res.tokens.push_back(ti.token);
    res.token_pos.emplace_back( token_pos_t{ti.begin, ti.end} );
  } while (ti.token!=tok_eof);
    
  return res;
}


//==============================================================================
// Lexer output operator
//==============================================================================
void print(std::ostream& os, const Tokens & toks, const lexer_results_t & res)
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
int Lexer::gettok(int & LastChar, std::string & IdentifierStr)
{

  auto NextChar = peek();
  IdentifierStr.clear();

  //----------------------------------------------------------------------------
  // identifier: [a-zA-Z][a-zA-Z0-9]*
  if (isalpha(LastChar)) {

    std::string str(1, LastChar);
    while (isalnum((LastChar = advance())) || LastChar=='_')
      str += LastChar;

    auto res = Tokens_.keywords.find(str);
    if (res!=tok_not_found) return res;
    
    IdentifierStr = str;
    return tok_identifier;
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
    //  LastChar = advance();
    //}

    // read first part of number
    bool is_float = (LastChar == '.');
    do {
      IdentifierStr += LastChar;
      LastChar = advance();
      if (LastChar == '.') {
        if (is_float)
          THROW_LEXER_ERROR( "Multiple '.' encountered in real", LexLoc_ );
        is_float = true;
        // eat '.'
        IdentifierStr += LastChar;
        LastChar = advance();
      }
    } while (isdigit(LastChar));

    if (LastChar == 'e' || LastChar == 'E') {
      is_float = true;
      // eat e/E
      IdentifierStr += LastChar;
      LastChar = advance();
      // make sure next character is sign or number
      if (LastChar != '+' && LastChar != '-' && !isdigit(LastChar))
        THROW_LEXER_ERROR( "Digit or +/- must follow exponent", LexLoc_ );
      // eat sign or number
      IdentifierStr += LastChar;
      LastChar = advance();
      // only numbers should follow
      do {
        IdentifierStr += LastChar;
        LastChar = advance();
      } while (isdigit(LastChar) );
    }

    if (is_float)
      return tok_real_literal;
    else
      return tok_int_literal;
  }

  //----------------------------------------------------------------------------
  // Comment until end of line.
  if (LastChar == tok_comment) {
    do
      LastChar = advance();
    while (LastChar != tok_eof && LastChar != '\n' && LastChar != '\r');

    return tok_comment;
  }

  //----------------------------------------------------------------------------
  // string literal
  if (LastChar == '\"') {
    std::string quoted;
    while ((LastChar = advance()) != '\"')
      quoted += LastChar;
    IdentifierStr = utils::unescape(quoted);
    LastChar = advance();
    return tok_string_literal;
  }
  
  //----------------------------------------------------------------------------
  // Comparison operators

  auto str = std::string(1,LastChar) + NextChar;
  auto tok = Tokens_.multi_char.find(str);

  if (tok != tok_not_found) {
    advance(); // eat next =
    LastChar = advance();
    return tok;
  }

  //----------------------------------------------------------------------------
  // Check for end of file.  Don't eat the EOF.
  if (LastChar == tok_eof)
    return tok_eof;

  //----------------------------------------------------------------------------
  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = advance();
  return ThisChar;
}

//==============================================================================
/// gettok - Return the next token from standard input.
//==============================================================================
token_info_t Lexer::gettok() {

  std::string identifier;
  int tok;
  std::ios::pos_type start_pos, end_pos;

  // Skip any whitespace.
  while (isspace(LastChar_))
    LastChar_ = advance();
  
  CurLoc_ = LexLoc_;

  start_pos = In_->tellg();
  tok = gettok(LastChar_, identifier);
  end_pos = In_->tellg();
  
  return {tok, start_pos, end_pos, identifier};
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
