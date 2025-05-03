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

} // namespace
