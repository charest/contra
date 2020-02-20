#include "token.hpp"

namespace contra {

//==============================================================================
// Get a tokens name
//==============================================================================
std::string getTokName(int Tok) {
  switch (Tok) {
  case tok_binary:
    return "binary";
  case tok_by:
    return "by";
  case tok_eof:
    return "eof";
  case tok_def:
    return "def";
  case tok_do:
    return "do";
  case tok_else:
    return "else";
  case tok_end:
    return "end";
  case tok_extern:
    return "extern";
  case tok_for:
    return "for";
  case tok_function:
    return "function";
  case tok_identifier:
    return "identifier";
  case tok_if:
    return "if";
  case tok_in:
    return "in";
  case tok_number:
    return "number";
  case tok_return:
    return "return";
  case tok_then:
    return "then";
  case tok_to:
    return "to";
  case tok_unary:
    return "unary";
  case tok_var:
    return "var";
  }
  return std::string(1, (char)Tok);
}

} // namespace
