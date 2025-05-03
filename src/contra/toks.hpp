#ifndef CONTRA_CONTRA_TOKS_HPP
#define CONTRA_CONTRA_TOKS_HPP

#include <string>


#define FOR_SYMBOLS(DO) \
  DO( TOK_ADD, '+' ) \
  DO( TOK_SUB, '-' ) \
  DO( TOK_MUL, '*' ) \
  DO( TOK_MOD, '%' ) \
  DO( TOK_EQ,  '=' )

#define FOR_KEYWORDS(DO) \
  DO( TOK_IF,      "if" ) \
  DO( TOK_ELIF,    "elif" ) \
  DO( TOK_ELSE,    "else" ) \
  DO( TOK_FOR,     "for" ) \
  DO( TOK_FOREACH, "foreach" ) \
  DO( TOK_BREAK,   "break" ) \
  DO( TOK_REDUCE,  "reduce" ) \
  DO( TOK_USE,     "use" ) \
  DO( TOK_TRUE,    "true" ) \
  DO( TOK_FALSE,   "false" ) \
  DO( TOK_FUNC,    "fn" ) \
  DO( TOK_RETURN,  "return" ) \
  DO( TOK_TASK,    "tsk" )

#define FOR_TYPES(DO) \
  DO( TOK_I64,      "i64" ) \
  DO( TOK_F64,      "f64" )
 
#define FOR_LEX_STATES(DO) \
  DO( TOK_IDENT,      "ident" ) \
  DO( TOK_INT_LIT,    "int_lit" ) \
  DO( TOK_REAL_LIT,   "real_lit" ) \
  DO( TOK_STRING_LIT, "string_lit" ) \
  DO( TOK_COMMENT,    "comment") \
  DO( TOK_ADD_EQ,     "+=" ) \
  DO( TOK_SUB_EQ,     "-=" ) \
  DO( TOK_MUL_EQ,     "*=" ) \
  DO( TOK_DIV_EQ,     "/=" ) \
  DO( TOK_EQUIV,      "==" ) \
  DO( TOK_NE,         "!=" ) \
  DO( TOK_GE,         ">=" ) \
  DO( TOK_LE,         "<=" ) \
  DO( TOK_UNK,        "unknown" ) \
  DO( TOK_EOF,        "eof" )
  
#define FOR_AST_NODES(DO) \
  DO( AST_FN_DEF,    "FunDef" ) \
  DO( AST_FN_CALL,   "FunCall" ) \
  DO( AST_FN_ARG,    "FunArg" ) \
  DO( AST_BLOCK,     "Block" ) \
  DO( AST_TSK_DEF,   "TskDef" ) \
  DO( AST_VAR,       "Var" ) \
  DO( AST_ARR_INIT,  "ArrInit" ) \
  DO( AST_ARR_INDEX, "ArrIndex" ) \
  DO( AST_IF,        "If") \
  DO( AST_FOR,       "For") \
  DO( AST_FOREACH,   "Foreach") \
  DO( AST_BREAK,     "Break") \
  DO( AST_LIT_REAL,  "RealLit") \
  DO( AST_LIT_INT,   "IntLit") \
  DO( AST_LIT_STRING,"StringLit") \
  DO( AST_REDUCE,    "Reduce") \
  DO( AST_REDUCE_OP, "ReduceOp") \
  DO( AST_USE,       "Use") \
  DO( AST_UNARY,     "Unary") \
  DO( AST_BINOP,     "Binary") \
  DO( AST_RANGE,     "Range") \
  DO( AST_ASSIGN,    "Assign") \
  DO( AST_EXPR_LIST, "ExprList") \
  DO( AST_RETURN,    "Return")


#define FOR_LEFT_ASSOC(DO) \
  DO( TOK_EQUIV, 1) \
  DO( TOK_NE,    1) \
  DO( '<',       2) \
  DO( TOK_LE,    2) \
  DO( '>',       2) \
  DO( TOK_GE,    2) \
  DO( '+',       3) \
  DO( '-',       3) \
  DO( '*',       4) \
  DO( '/',       4) \
  DO( '%',       4)

#define FOR_RIGHT_ASSOC(DO)

#define FOR_UNARY(DO) \
  DO( '+', 5) \
  DO( '-', 5)

namespace contra {

enum Toks {
#define DEFINE_TOKS(name, ch, ...) name = ch,
  FOR_SYMBOLS   (DEFINE_TOKS)
#undef DEFINE_TOKS
  _TOKS_START_  = 255,
#define DEFINE_TOKS(name, str, ...) name,
  FOR_LEX_STATES(DEFINE_TOKS)
  FOR_KEYWORDS  (DEFINE_TOKS)
  FOR_TYPES     (DEFINE_TOKS)
  FOR_AST_NODES (DEFINE_TOKS)
#undef DEFINE_TOKS
};

/// convert the ast type to a string
inline std::string tok_to_string(int ty)
{
#define DEFINE_TOKS(name, str, ...) case name: return str;
  switch (ty) {
  FOR_LEX_STATES(DEFINE_TOKS)
  FOR_KEYWORDS  (DEFINE_TOKS)
  FOR_TYPES     (DEFINE_TOKS)
  FOR_AST_NODES (DEFINE_TOKS)
  case 0 ... 255:  return std::string(1, ty);
  }
  return "Error";
#undef DEFINE_TOKS
}
  

inline bool tok_is_type(int ty)
{
#define DEFINE_TOKS(name, str, ...) case name: return true;
  switch (ty) {
  FOR_TYPES(DEFINE_TOKS)
  default: return false;
  };
}


struct BinopPrecedence;
struct token_map_t;
  
/// Main token builder 
token_map_t make_contra_tokens();
token_map_t make_sext_tokens();

/// Main precedence builder
BinopPrecedence make_contra_precedence();

} // namespace

#endif // CONTRA_CONTRA_TOKS_HPP
