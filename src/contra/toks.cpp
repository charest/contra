#include "toks.hpp"
#include "token.hpp"
#include "precedence.hpp"

namespace contra {

//==============================================================================
// Make contra tokens
//==============================================================================
token_map_t make_contra_tokens() {
  token_map_t toks;

  #define INSTALL_TOKS(name, str, ...) \
    toks.add( name, str);

  FOR_KEYWORDS  (INSTALL_TOKS)
  FOR_TYPES     (INSTALL_TOKS)

  #undef INSTALL_TOKS
  
  return toks;
}

//==============================================================================
// Make sext tokens
//==============================================================================
token_map_t make_sext_tokens() {
  token_map_t toks;

  #define INSTALL_TOKS(name, str, ...) \
    toks.add( name, str);

  FOR_AST_NODES (INSTALL_TOKS)
  
  #undef INSTALL_TOKS
  
  return toks;
}


//==============================================================================
// Make contra precedence
//==============================================================================
BinopPrecedence make_contra_precedence() {
  BinopPrecedence p;
 
  #define INSTALL(name, prec) \
    p.binary_left[name] = prec;

  FOR_LEFT_ASSOC(INSTALL)

  #undef INSTALL
  
  #define INSTALL(name, prec) \
    prec.binary_right[name] = prec;

  FOR_RIGHT_ASSOC(INSTALL)
  
  #undef INSTALL
  
  #define INSTALL(name, prec) \
    p.unary[name] = prec;

  FOR_UNARY(INSTALL)
  
  #undef INSTALL

  return p;
}

#if 0
static std::string lex_to_str(int tok)
{
  switch (tok) {
  case LEX_UNK:    return "UNK";
  case LEX_IDENT:  return "IDENT";
  case LEX_INT:    return "INT";
  case LEX_REAL:   return "REAL";
  case LEX_COMMENT:return "COMMENT";
  case LEX_QUOTED: return "QUOTED";
  case LEX_ADD_EQ: return "ADD_EQ";
  case LEX_SUB_EQ: return "SUB_EQ";
  case LEX_MUL_EQ: return "MUL_EQ";
  case LEX_DIV_EQ: return "DIV_EQ";
  case LEX_EQUIV:  return "EQUIV";
  case LEX_NE:     return "NE";
  case LEX_GE:     return "GE";
  case LEX_LE:     return "LE";
  case LEX_EOF:    return "EOF";
  case 0 ... 255:  return std::string(1, tok);
  default:         return "Error";
  };
}
#endif

} // namespace
