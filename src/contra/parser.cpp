#include "errors.hpp"
#include "graph.hpp"
#include "identifier.hpp"
#include "parser.hpp"
#include "precedence.hpp"
#include "stream.hpp"
#include "toks.hpp"
#include "token.hpp"

#include "utils/string_utils.hpp"

#include <iomanip>
#include <list>
#include <map>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

namespace contra {

struct parse_res_t
{
  int err=0, node=-1;
  parse_res_t & operator+=(const parse_res_t & o)
  { err += o.err; return *this; }
  
  parse_res_t & operator+=(int err)
  { err += err; return *this; }
};

//==============================================================================
/// Dump parser results in tabular form
//==============================================================================
void print(
  std::ostream& os,
  const parse_tree_t & tree,
  const graph_t & graph
)
{
  auto n = tree.size();

  using utils::printRight, utils::printLeft;
  int digits = utils::count_digits(n);
  auto nw = std::max(digits+1, 6);
  auto pw = std::max(digits+1, 8);
  auto tw = 6;
  auto sw = 14;
  auto cw = 4*nw;
  auto kw = std::max(digits+1, 7);

  printRight(os, nw, ' ', "NodeId");
  printRight(os, 2, ' ');
  printRight(os, tw, ' ', "TypeId");
  printRight(os, 2, ' ');
  printRight(os, sw, ' ', "TypeStr");
  printRight(os, 2, ' ');
  printRight(os, kw, ' ', "TokenId");
  printRight(os, 2, ' ');
  printRight(os, pw, ' ', "ParentId");
  printRight(os, 2, ' ');
  printLeft (os, cw, ' ', "ChildIds");
  os << std::endl;

  printRight(os, nw, '-');
  printRight(os, 2, ' ');
  printRight(os, tw, '-');
  printRight(os, 2, ' ');
  printRight(os, sw, '-');
  printRight(os, 2, ' ');
  printRight(os, kw, '-');
  printRight(os, 2, ' ');
  printRight(os, pw, '-');
  printRight(os, 2, ' ');
  printLeft (os, cw, '-');
  os << std::endl;
  
  for (int i=0; i<n; ++i) {
    std::stringstream ss;
    auto nc = graph.size(i);
    if (nc) {
      for (int c=0; c<nc-1; ++c) ss << graph(i,c) << ",";
      if (nc>0) ss << graph(i,nc-1);
    }

    auto ty = tree.node_ast_type[i];
    auto tid = tree.node_to_token[i];
    printRight(os, nw, ' ', i);
    printRight(os, 2, ' ');
    printRight(os, tw, ' ', ty);
    printRight(os, 2, ' ');
    printRight(os, sw, ' ', tok_to_string(ty));
    printRight(os, 2, ' ');
    printRight(os, kw, ' ', tid);
    printRight(os, 2, ' ');
    printRight(os, pw, ' ', tree.node_parent[i]);
    printRight(os, 2, ' ');
    printLeft (os, cw, ' ', ss.str());
    os << std::endl;
  }
}

//==============================================================================
/// Dump parser results in sext form
//==============================================================================
void print(
  std::ostream& os,
  const token_map_t & toks,
  const lexed_t & lex,
  const parse_tree_t & tree,
  const graph_t & graph)
{
  auto n = tree.size();

  std::stack<std::pair<int,int>> q;
  bool first = true;
  
  for (auto r : graph.roots) q.push({r,0});
    
  while (q.size()) {
    auto curr = q.top();
    auto i = curr.first;
    auto depth = curr.second;
    q.pop();
    
    if (depth == -1) {
      os << ")";
      continue;
    }

    auto ty = tree.node_ast_type[i];
    auto tid = tree.node_to_token[i];
    auto tok = lex.tokens[tid];
    
    auto space = std::string(2*depth, ' ');
    if (first) first = false;
    else       os << std::endl;
    os << space << "(" << tok_to_string(ty);

    switch (ty) {
    case AST_FN_DEF:
    case AST_FN_CALL:
    case AST_VAR:
    case AST_ARR_INDEX:
    case AST_LIT_REAL:
    case AST_LIT_INT:
    case AST_LIT_STRING:
    case AST_ARR: {
      auto id = lex.findIdentifier(tid);
      os << " " << lex.getIdentifierString(id);
      break;
    }
    
    case AST_ASSIGN:
    case AST_UNARY:
    case AST_BINOP: {
      os << " " << tok_to_string(tok);
      break;
    }

    case AST_REDUCE_OP: {
      if (tok == TOK_IDENT) {
        auto id = lex.findIdentifier(tid);
        os << " " << lex.getIdentifierString(id);
      }
      else {
        os << tok_to_string(tok);
      }
      break;
    }
    }

    q.push({i, -1});
    auto nc = graph.size(i);
    for (int c=nc; c-->0; ) q.push({graph(i,c), depth+1});
  }
  os << std::endl;

}
    

//==============================================================================
/// Compare two ast tree nodes
//==============================================================================
bool compare_ast_node(
  const token_map_t & toka,
  const lexed_t & lxa,
  const parse_tree_t & tra,
  const token_map_t & tokb,
  const lexed_t & lxb,
  const parse_tree_t & trb,
  int na,
  int nb)
{
  // ast types
  auto ast_tya = tra.node_ast_type[na];
  auto ast_tyb = trb.node_ast_type[nb];
  if (ast_tya != ast_tyb) {
    std::cerr << "Node types differ : ";
    std::cerr << "{" << na << ", " << tok_to_string(ast_tya);
    std::cerr << "} vs {";
    std::cerr << nb << ", " << tok_to_string(ast_tyb) << "}";
    std::cerr << std::endl;
    return false;
  }
  
  // token types
  auto ta = tra.node_to_token[na];
  auto tb = trb.node_to_token[nb];
  auto tya = lxa.tokens[ta];
  auto tyb = lxb.tokens[tb];
  auto ida = lxa.findIdentifier(ta);
  auto idb = lxb.findIdentifier(tb);
  auto stra = lxa.getIdentifierString(ida);
  auto strb = lxb.getIdentifierString(idb);

  switch (ast_tya) {

  case (AST_UNARY):
  case (AST_BINOP):
  case (AST_ASSIGN):
    if (tya != tyb) {
      std::cerr << "Operators don't match. ";
      std::cerr << "{" << na << ", " << tok_to_string(tya);
      std::cerr << "} vs {";
      std::cerr << nb << ", " << tok_to_string(tyb) << "}";
      std::cerr << std::endl;
      return false;
    }
    break;
  
  case (AST_FN_CALL):
  case (AST_VAR):
  case (AST_ARR_INDEX):
  case (AST_LIT_INT):
  case (AST_LIT_REAL):
  case (AST_LIT_STRING):
  
    if (ida == -1 || idb == -1) {
      std::cerr << "Expected identifiers in both. ";
      std::cerr << std::endl;
      std::cerr << "{" << na << ", " << stra;
      std::cerr << "} vs {";
      std::cerr << nb << ", " << strb << "}";
      std::cerr << std::endl;
      return false;
    }
    if (stra != strb) {
      std::cerr << "Identifiers don't match. ";
      std::cerr << std::endl;
      std::cerr << "{" << na << ", " << stra;
      std::cerr << "} vs {";
      std::cerr << nb << ", " << strb << "}";
      std::cerr << std::endl;
      return false;
    }
    
    break;
  
  case (AST_REDUCE_OP):

    if (tya != tyb || stra != strb) {
      std::cerr << "Operators or identifiers don't match. ";
      std::cerr << "{" << na << ", " << tok_to_string(tya) << ", " << stra;
      std::cerr << "} vs {";
      std::cerr << nb << ", " << tok_to_string(tyb) << ", " << strb << "}";
      std::cerr << std::endl;
      return false;
    }
    break;

  }

  return true;
}

//==============================================================================
/// Compare two trees
//==============================================================================
bool compare(
  stream_t & isa,
  const token_map_t & toka,
  const lexed_t & lxa,
  const parse_tree_t & tra,
  const graph_t & gra,
  stream_t & isb,
  const token_map_t & tokb,
  const lexed_t & lxb,
  const parse_tree_t & trb,
  const graph_t & grb)
{

  auto nr = gra.roots.size();
  if (nr != grb.roots.size()) {
    std::cerr << "Roots: " << nr << " vs " << grb.roots.size() << std::endl;
    return false;
  }

  std::stack<std::pair<int,int>> q;
  for (int i=0; i<nr; ++i)
    q.push({gra.roots[i], grb.roots[i]});
    
  while (q.size()) {
    auto curr = q.top();
    auto ia = curr.first;
    auto ib = curr.second;
    q.pop();

    if (!compare_ast_node(toka, lxa, tra, tokb, lxb, trb, ia, ib))
    {
      auto ta = tra.node_to_token[ia];
      auto tb = trb.node_to_token[ib];
      error(isa, "Left tree is:", lxa.token_pos[ta]);
      error(isb, "Right tree is:", lxb.token_pos[tb]);
      return false;
    }


    // children
    auto na = gra.size(ia);
    auto nb = grb.size(ib);
    if (na != nb) {
      auto ta = tra.node_to_token[ia];
      auto tb = trb.node_to_token[ib];
      std::cerr << "Number of children differ: " << na << " vs " << nb << std::endl;
      error(isa, "Left tree is:", lxa.token_pos[ta]);
      error(isb, "Right tree is:", lxb.token_pos[tb]);
      return false;
    }

    for (int c=na; c-->0; )
      q.push({gra(ia,c), grb(ib,c)});
  }
  
  return true;
}
  
void consume(size_t ntok, int & tok)
{
  tok++;
  if (tok >= ntok) 
    throw std::runtime_error("Ran out of tokens!");
}
  
int consume(
  stream_t & is,
  const lexed_t & lx,
  int & tok,
  int c,
  const char * msg)
{
  int err = 0;
  if (lx.tokens[tok] != c)
    err += error(is, msg, lx.token_pos[tok]);
  tok++;
  return err;
}

//==============================================================================
parse_res_t parse_primary_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok);

parse_res_t parse_binop_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok,
  int MinPrec);

//==============================================================================
// unary
//   ::= primary
//   ::= '!' unary
//==============================================================================
parse_res_t parse_unary_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto ty = lx.tokens[tok];
  auto Prec = prec.findUnary(ty);

  // If this is a unary operator, read it.
  if (Prec != -1) {
    auto new_node = tree.addNode(tok, AST_UNARY, parent);
    consume(lx.tokens.size(), tok);
    auto ret = parse_binop_expr(new_node, is, lx, prec, tree, tok, Prec);
    return {ret.err, new_node};
  }
  // If the current token is not an operator, it must be a primary expr.
  else if (prec.findBinary(ty) == -1)
  {
    return parse_primary_expr(parent, is, lx, prec, tree, tok);
  }
  // unknown unary operator.  Give back the parent in hopes of recovery
  else {
    return { error(is, "Unknown unary operator.", lx.token_pos[tok]), parent };
  }
}

//==============================================================================
// binoprhs
//   ::= ('+' primary)*
//==============================================================================
parse_res_t parse_binop_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok,
  int MinPrec)
{
  auto lhs = parse_unary_expr(parent, is, lx, prec, tree, tok);

  // If this is a binop, find its precedence.
  while (true) {
  
    // check left precedence
    auto op = lx.tokens[tok];
    auto TokPrec = prec.findLeft(op);
    bool isLeft = (TokPrec != -1);
    // otherwise, check right
    if (!isLeft) TokPrec = prec.findRight(op);
    
    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < MinPrec) break;

    // Okay, we know this is a binop.
    auto binop = tree.addNode(tok, AST_BINOP, parent);
    tree.setParent(lhs.node, binop); // repoint the left
    
    consume(lx.tokens.size(), tok); // eat binop
    
    // Parse the unary expression after the binary operator.
    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    auto NextPrec = isLeft ? TokPrec+1 : TokPrec;
    lhs += parse_binop_expr(binop, is, lx, prec, tree, tok, NextPrec);
    
    // Move the binop to the left
    lhs.node = binop;

  }
  
  return lhs;
}



//==============================================================================
// expression
//   ::= primary binoprhs , primary binoprhs
//   ::= primary binoprhs : primary binoprhs
//   ::= primary binoprhs = primary binoprhs
//==============================================================================
parse_res_t parse_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto & tokens = lx.tokens;

  // parse first expr
  auto [err, lhs] = parse_binop_expr(parent, is, lx, prec, tree, tok, 0);
  
  // hit a list
  if (tokens[tok] == ',') {
    // create new root node unless parent is a call
    auto expr_list = tree.node_ast_type[parent] == AST_FN_CALL ? 
      parent : tree.addNode(tok, AST_EXPR_LIST, parent);
    // repoint the old root
    tree.setParent(lhs, expr_list);
    // add to the list
    while (tokens[tok] == ',') {
      consume(lx.tokens.size(), tok); // eat ,
      err += parse_binop_expr(expr_list, is, lx, prec, tree, tok, 0).err;
    }
    // set new lhs
    lhs = expr_list;
  }
  else if (tokens[tok] == ':') {
    // create new root node
    auto range_expr = tree.addNode(tok, AST_RANGE, parent);
    // repoint the old root
    tree.setParent(lhs, range_expr);
    // add to the list
    int num_exprs = 1;
    while (tokens[tok] == ':') {
      consume(lx.tokens.size(), tok); // eat :
      err += parse_binop_expr(range_expr, is, lx, prec, tree, tok, 0).err;
      num_exprs++;
    }
    // validate the number of expressions found
    if (num_exprs > 3 || num_exprs < 2) {
      err += error(
        is, "Only 'begin':'end':['step'] specification supported for ranges." ,
        lx.token_pos[tok]);
    }
    // set new lhs
    lhs = range_expr;
  }
  
  if (tokens[tok] == '=') {
    // create a new root node
    auto assign_expr = tree.addNode(tok, AST_ASSIGN, parent);
    consume(lx.tokens.size(), tok); // eat =
    // repoint the old root
    tree.setParent(lhs, assign_expr);
    // parse the rhs
    err += parse_expr(assign_expr, is, lx, prec, tree, tok).err;
    // set new lhs
    lhs = assign_expr;
  }

  return {err, lhs};
}

//==============================================================================
// numberexpr ::= number
// breakexpr
//==============================================================================
// BreakStmtAST - ast_break
// ValueExprAST - ast_lit_int, ast_lit_real, ast_lit_string
template<int Ty>
parse_res_t parse_simple_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  parse_tree_t & tree,
  int & tok)
{
  // add token
  auto node = tree.addNode(tok, Ty, parent);
  consume(lx.tokens.size(), tok);
  return {0, node};
}

//==============================================================================
// Return expression
//==============================================================================
parse_res_t parse_return(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{ 
  auto node = tree.addNode(tok, AST_RETURN, parent);

  // eat return
  consume(lx.tokens.size(), tok);
  
  return parse_expr(node, is, lx, prec, tree, tok);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
parse_res_t parse_parens_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{ 
  // eat (
  consume(lx.tokens.size(), tok);
  // add expression in parens, parent passes through
  auto res = parse_expr(parent, is, lx, prec, tree, tok);
  // eat )
  res += consume(is, lx, tok, ')', "Expected ')' after expression");
  return res;
}


//==============================================================================
// identifierexpr
//   ::= 
//   ::= identifier '(' expression* ')'
//==============================================================================
parse_res_t parse_identifier_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  // store identifier token
  auto first_tok = tok;
  
  auto & tokens = lx.tokens;
  
  consume(tokens.size(), tok); // eat identifier.

  //----------------------------------------------------------------------------
  // Call.
  if (tokens[tok] == '(') {
    auto node = tree.addNode(first_tok, AST_FN_CALL, parent);
    parse_res_t ret{0, node};
    consume(lx.tokens.size(), tok); // eat (
    if (tokens[tok] != ')')
      ret += parse_expr(node, is, lx, prec, tree, tok);
    ret += consume(is, lx, tok, ')', "Expected ')'."); // Eat the ')'.
    return ret;
  }

  //----------------------------------------------------------------------------
  // Variable reference
  else {

    parse_res_t ret;
      
    // Has a type, so we know its a decl
    bool has_type = false;
    int type_tok = -1;
    auto ident_tok = first_tok;

    auto tok_ty = tokens[first_tok];
    if (tok_is_type(tok_ty)) 
    {
      ident_tok = tok;
      type_tok = first_tok;
      has_type = true;
      ret += consume(is, lx, tok, TOK_IDENT, "Expected identifier."); // eat the identifier
    }
    else if (tok_ty != TOK_IDENT) 
      ret += error(is, "Expected identifier or type.", lx.token_pos[first_tok]);

    //----------------------------------
    // Array
    if (tokens[tok] == '[') {
      ret.node = tree.addNode(ident_tok, AST_ARR_INDEX, parent);
      consume(lx.tokens.size(), tok); // eat [
      ret += parse_expr(ret.node, is, lx, prec, tree, tok);
      ret += consume(is, lx, tok, ']', "Expected ']'  in array access/declaration."); // eat ]
    }
    
    //----------------------------------
    // scalar
    else {
      ret.node = tree.addNode(ident_tok, AST_VAR, parent);
    }

    // add type info if any
    if (has_type) {
      auto ty = tree.installType(tokens[type_tok]);
      tree.setType(ret.node, ty, type_tok);
    }

    return ret;

  } // variable reference

}
    

//==============================================================================
// Parse basic block until a token is found
//==============================================================================
int parse_until(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok,
  int to_tok)
{
  int err = 0;
  while (lx.tokens[tok] != to_tok) {
    err += parse_expr(parent, is, lx, prec, tree, tok).err;
    if (lx.tokens[tok] == ';') consume(lx.tokens.size(), tok);
  }
  return err;
}

//==============================================================================
// ifexpr ::= 'if' expression 'then' expression 'else' expression
//==============================================================================
parse_res_t parse_if_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto & tokens = lx.tokens;
  
  auto if_node = tree.addNode(tok, AST_IF, parent);
  parse_res_t ret{0, if_node};

  // TODO if, elseif, else is repeated
  // TODO drop semicolons in lexer

  //---------------------------------------------------------------------------
  // If
  {

    consume(tokens.size(), tok); // eat the if.

    // 1 - condition.
    auto cond_node = tree.addNode(tok, AST_IF_COND, if_node);
    ret += parse_expr(cond_node, is, lx, prec, tree, tok);
    
    // 2 - body.
    auto body_node = tree.addNode(tok, AST_IF_BODY, if_node);
      
    //------------------------------------
    // Multi-liner TODO LIFT OUT BASIC BLOCK
    if (tokens[tok] == '{') {
      consume(tokens.size(), tok); // eat {
      ret.err += parse_until(body_node, is, lx, prec, tree, tok, '}');
      consume(tokens.size(), tok); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      ret += parse_expr(body_node, is, lx, prec, tree, tok);
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (tokens[tok] == TOK_ELIF) {
  
    consume(tokens.size(), tok); // eat elif

    // 1 - condition.
    auto cond_node = tree.addNode(tok, AST_ELIF_COND, if_node);
    ret += parse_expr(cond_node, is, lx, prec, tree, tok);

    // 2 - body
    auto body_node = tree.addNode(tok, AST_ELIF_BODY, if_node);
  
    //------------------------------------
    // Multi-liner
    if (tokens[tok] == '{') {
      consume(tokens.size(), tok); // eat {
      ret.err += parse_until(body_node, is, lx, prec, tree, tok, '}');
      consume(tokens.size(), tok); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      ret += parse_expr(body_node, is, lx, prec, tree, tok);
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (tokens[tok] == TOK_ELSE) {

    consume(tokens.size(), tok); // eat else
    
    auto body_node = tree.addNode(tok, AST_ELSE_BODY, if_node);
    
    //------------------------------------
    // Multi-liner
    if (tokens[tok] == '{') {
      consume(tokens.size(), tok); // eat {
      ret.err += parse_until(body_node, is, lx, prec, tree, tok, '}');
      consume(tokens.size(), tok); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      ret += parse_expr(body_node, is, lx, prec, tree, tok);
    }

  }

  return ret;
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
parse_res_t parse_for_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto & tokens = lx.tokens;

  int ast_for_type = (tokens[tok] == TOK_FOREACH) ? AST_FOREACH : AST_FOR;
 
  // Top for node
  auto for_node = tree.addNode(tok, ast_for_type, parent);
  parse_res_t ret{0, for_node};

  consume(tokens.size(), tok); // eat the for.

  if (tokens[tok] != TOK_IDENT)
    ret += error(is, "Expected identifier after 'for'", lx.token_pos[tok]);

  // variable node
  tree.addNode(tok, AST_VAR, for_node);
  
  consume(tokens.size(), tok); // eat identifier.

  ret += consume(is, lx, tok, '=', "Expected '=' after 'for'"); // eat =
  
  // range node
  ret += parse_expr(for_node, is, lx, prec, tree, tok);

  // body node
  auto body_node = tree.addNode(tok, AST_BLOCK, for_node);

  //------------------------------------
  // Multi-liner
  if (tokens[tok] == '{') {
    consume(tokens.size(), tok); // eat {
    ret.err += parse_until(body_node, is, lx, prec, tree, tok, '}');
    consume(tokens.size(), tok); // eat }
  }
  //------------------------------------
  // One-liner
  else {
    ret += parse_expr(body_node, is, lx, prec, tree, tok);
  }

  return ret;
  
}

//==============================================================================
// Array expression parser
//==============================================================================
parse_res_t parse_array_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{

  auto & tokens = lx.tokens;

  auto arr_node = tree.addNode(tok, AST_ARR, parent);
  parse_res_t ret{0, arr_node};

  consume(tokens.size(), tok); // eat [.

  ret += parse_expr(arr_node, is, lx, prec, tree, tok);
    
  if (tokens[tok] == ';') {
    consume(tokens.size(), tok); // eat ;
    ret += parse_expr(arr_node, is, lx, prec, tree, tok);
  }

  ret += consume(is, lx, tok, ']', "Expected ']'"); // eat ]

  return ret;

}

//==============================================================================
// reduction
//==============================================================================
parse_res_t parse_reduce_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto & tokens = lx.tokens;
  int err = 0;
  
  auto node = tree.addNode(tok, AST_REDUCE, parent);

  consume(tokens.size(), tok);;  // eat the reduce
    
  while (tokens[tok] != ':') {
    if (tokens[tok] != TOK_IDENT)
      err += error(is, "Expected an identifier after keyword 'reduce'.", lx.token_pos[tok]);
    tree.addNode(tok, AST_VAR, node);  
    consume(tokens.size(), tok); // eat identifier.
    if (tokens[tok] == ',') consume(tokens.size(), tok); // eat ,
  }

  err += consume(is, lx, tok, ':', "Expected ':'."); // eat ":".

  auto ty = tokens[tok];
  if ( (prec.findBinary(ty) == -1) && (ty != TOK_IDENT) )
    err += error(is, "Expected identifier or operator after ':'.", lx.token_pos[tok]);

  tree.addNode(tok, AST_REDUCE_OP, node);

  consume(tokens.size(), tok);; // eat identifier

  return {err, node};

}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
parse_res_t parse_part_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto & tokens = lx.tokens;
  int err = 0;

  auto node = tree.addNode(tok, AST_USE, parent);
    
  consume(tokens.size(), tok);  // eat the use
  
  while (tokens[tok] != ':') {
    if (tokens[tok] != TOK_IDENT)
      err += error(is, "Expected an identifier after keyword 'use'.", lx.token_pos[tok]);
    tree.addNode(tok, AST_VAR, node);  
    consume(tokens.size(), tok); // eat identifier.
    if (tokens[tok] == ',') consume(tokens.size(), tok); // eat ,
  }

  err += consume(is, lx, tok, ':', "Expected ':'."); // eat ":".

  if (tokens[tok] != TOK_IDENT)
    err += error(is, "Expected identifier after ':'.", lx.token_pos[tok]);
    
  auto [err2, child] = parse_expr(node, is, lx, prec, tree, tok);

  return {err+err2, node};
}

//==============================================================================
// primary
//   ::= identifierexpr
//   ::= numberexpr
//   ::= parenexpr
//   ::= ifexpr
//   ::= forexpr
//   ::= varexpr
//==============================================================================
parse_res_t parse_primary_expr(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  switch (lx.tokens[tok]) {
  case TOK_IDENT:
  case TOK_I64:
  case TOK_F64:
    return parse_identifier_expr(parent, is, lx, prec, tree, tok);
  case TOK_REAL_LIT:
    return parse_simple_expr<AST_LIT_REAL>(parent, is, lx, tree, tok);
  case TOK_INT_LIT:
    return parse_simple_expr<AST_LIT_INT>(parent, is, lx, tree, tok);
  case '(':
    return parse_parens_expr(parent, is, lx, prec, tree, tok);
  case '[':
    return parse_array_expr(parent, is, lx, prec, tree, tok);
  case TOK_IF:
    return parse_if_expr(parent, is, lx, prec, tree, tok);
  case TOK_FOR:
  case TOK_FOREACH:
    return parse_for_expr(parent, is, lx, prec, tree, tok);
  case TOK_USE:
    return parse_part_expr(parent, is, lx, prec, tree, tok);
  case TOK_REDUCE:
    return parse_reduce_expr(parent, is, lx, prec, tree, tok);
  case TOK_STRING_LIT:
    return parse_simple_expr<AST_LIT_STRING>(parent, is, lx, tree, tok);
  case TOK_BREAK:
    return parse_simple_expr<AST_BREAK>(parent, is, lx, tree, tok);
  case TOK_RETURN:
    return parse_return(parent, is, lx, prec, tree, tok);
  }
      
  // return parent and hope for recovery
  auto old_tok = tok;
  consume(lx.tokens.size(), tok);
  return {
    error(is, "Unknown token when expecting an expression", lx.token_pos[old_tok]),
    parent
  };
}

#if 0
//==============================================================================
// prototype
//==============================================================================
parse_res_t parse_prototype(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{

  int err = 0;
  auto & tokens = lx.tokens;
  auto ntok = tokens.size();

  std::vector<int> ReturnTypes;
  int proto = -1;

  // know it has specified type
  if (isType(tokens[tok])) {

    while ((tokens[tok]!=tok_ident) && (tok<ntok)) {
      auto tok_ty = tokens[tok];
      if (!isType(tok_ty))
        err += error(is, "Expected type.", lx.token_pos[tok]);
      auto ty = tree.installType(tok_ty);
      ReturnTypes.emplace_back(ty);
      ++tok; // eat type
      if (tokens[tok] == ',') ++tok; // eat comma
    }

    if (tokens[tok] != tok_ident)
      err += error(is, "Expected function name specification.", lx.token_pos[tok]);
    
    // create the node
    proto = tree.addNode(parent, AST_FN_DEF, tok);
    
    // add final type and set node
    auto fun_ty = tree.installType( ReturnTypes );  
    tree.setType(proto, fun_ty);

    ++tok; // eat identifier
  }
  // no specified type
  else {
    proto = tree.addNode(parent, AST_FN_DEF, first_tok);
  }

  
  if (tokens[tok] != '(')
    err += error(is, "Expected '(' in prototype", lx.token_pos[tok]);

  ++tok; // eat "("

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  while ((tokens[tok]!=')') && (tok<ntok)) {

    bool IsArray = false;
    auto type_tok = tok;

    // token must be a type
    if (!isType(tokens[tok]))
      err += error(is, "Type expected.", lx.token_pos[tok]);

    ++tok; // eat type
  
    if (tokens[tok] != tok_ident)
      err += error(is, "Mising type or variable name in prototype.", lx.token_pos[tok]);
   
    auto arg = tree.addNode(proto, ast_var, tok);
    auto arg_ty = tree.installType(type_tok);
    tree.setType(arg, arg_ty, type_tok);
    
    ++tok; // eat identifier
    
    if (tokens[tok] == '[') {
      IsArray = true;
      ++tok; // eat the '['.
      if (tokens[tok] != ']')
        err += error(is, "Expected ']'", lx.token_pos[tok]);
      ++tok; // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (tokens[tok] == ',') ++tok; // eat ','
  }

  if (CurTok_ != ')')
    THROW_SYNTAX_ERROR(
        "Expected ')' in prototype",
        getIdentifierLoc());

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && Args.size() != Kind)
    THROW_SYNTAX_ERROR(
        "Invalid number of operands for operator: "
        << Kind << " expected, but got " << Args.size(),
        getIdentifierLoc());

  return std::make_unique<PrototypeAST>(
      Identifier{FnName, FnLoc},
      std::move(Args),
      std::move(ArgTypes),
      std::move(ArgIsArray),
      std::move(ReturnTypes),
      Kind != 0,
      BinaryPrecedence);
}
#endif


//==============================================================================
// Toplevel function parser
//==============================================================================
parse_res_t parse_function(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  int err = 0;

  const auto & tokens = lx.tokens;
  
  bool IsTask = (tokens[tok] == TOK_TASK);
  auto ast_ty = IsTask ? AST_TSK_DEF : AST_FN_DEF;

  consume(tokens.size(), tok);; // eat 'function' / 'task'

  //---------------------------------------------------------------------------
  // Return types
  
  std::vector<int> ReturnTypes;
  int fn_node = -1;

  // know it has specified type
  if (tok_is_type(tokens[tok])) {

    while (tokens[tok] != TOK_IDENT) {
      auto tok_ty = tokens[tok];
      if (!tok_is_type(tok_ty))
        err += error(is, "Expected type.", lx.token_pos[tok]);
      auto ty = tree.installType(tok_ty);
      ReturnTypes.emplace_back(ty);
      consume(tokens.size(), tok); // eat type
      if (tokens[tok] == ',') consume(tokens.size(), tok); // eat comma
    }

    if (tokens[tok] != TOK_IDENT)
      err += error(is, "Expected function name specification.", lx.token_pos[tok]);
    
    // create the node
    fn_node = tree.addNode(tok, ast_ty, parent);
    
    // add final type and set node
    auto fun_ty = tree.installType( ReturnTypes );  
    tree.setType(fn_node, fun_ty);

  }
  // no specified type
  else {
    fn_node = tree.addNode(tok, ast_ty, parent);
  }

  consume(tokens.size(), tok); // eat identifier
  
  err += consume(is, lx, tok, '(', "Expected '(' in prototype"); // eat "("
  
  //---------------------------------------------------------------------------
  // Arguments

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  auto args_node = tree.addNode(tok, AST_FN_ARGS, fn_node);

  while (tokens[tok] != ')') {

    bool IsArray = false;
    auto type_tok = tok;

    // token must be a type
    if (!tok_is_type(tokens[tok]))
      err += error(is, "Type expected.", lx.token_pos[tok]);

    consume(tokens.size(), tok); // eat type
  
    if (tokens[tok] != TOK_IDENT)
      err += error(is, "Mising type or variable name in prototype.", lx.token_pos[tok]);
   
    auto arg = tree.addNode(tok, AST_VAR, args_node);
    auto arg_ty = tree.installType(type_tok);
    tree.setType(arg, arg_ty, type_tok);
    
    consume(tokens.size(), tok); // eat identifier
    
    if (tokens[tok] == '[') {
      IsArray = true;
      consume(tokens.size(), tok); // eat the '['.
      err += consume(is, lx, tok, ']', "Expected ']'"); // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (tokens[tok] == ',') consume(tokens.size(), tok); // eat ','
  }

  if (tokens[tok] != ')')
    err += error(is, "Expected ')' in prototype", lx.token_pos[tok]);

  // success.
  consume(tokens.size(), tok); // eat ')'.

#if 0
  // Verify right number of names for operator.
  if (Kind && Args.size() != Kind)
    THROW_SYNTAX_ERROR(
        "Invalid number of operands for operator: "
        << Kind << " expected, but got " << Args.size(),
        getIdentifierLoc());

  return std::make_unique<PrototypeAST>(
      Identifier{FnName, FnLoc},
      std::move(Args),
      std::move(ArgTypes),
      std::move(ArgIsArray),
      std::move(ReturnTypes),
      Kind != 0,
      BinaryPrecedence);
#endif

  //---------------------------------------------------------------------------
  // Function body

  //------------------------------------
  // Multi-liner
  if (tokens[tok] == '{') {
    auto body_node = tree.addNode(tok, AST_BLOCK, fn_node);
    consume(tokens.size(), tok); // eat {
    parse_until(body_node, is, lx, prec, tree, tok, '}');
    err += consume(is, lx, tok, '}', "Expected '}'."); // eat }
  }
  //------------------------------------
  // One-liner
  else {
    err += parse_expr(fn_node, is, lx, prec, tree, tok).err;
  }

  return {err, fn_node};
}

//==============================================================================
// Main parse function
//==============================================================================
int parse(
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree)
{
  auto ntokens = lx.tokens.size();
  int i=0;
  int err = 0;

  while (i < ntokens) {

    auto tok = lx.tokens[i];

    switch (tok) {
    case TOK_EOF:
      goto exit_loop; // don't hate
    case ';':
      ++i;
      break;
    case TOK_TASK:
    case TOK_FUNC:
      err += parse_function(-1, is, lx, prec, tree, i).err;
      break;
    default:
      err += parse_expr(-1, is, lx, prec, tree, i).err;

    }

  }
  exit_loop:;

  return err;
}


//==============================================================================
// Parse a node
//==============================================================================
parse_res_t parse_sext_node(
  int parent,
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto & tokens = lx.tokens;
  
  int node = -1;
  int err = 0;

  auto ast_tok = tok;
  auto ast_ty = tokens[tok];
  auto ast_pos = lx.token_pos[tok];
  consume(tokens.size(), tok);
  auto tok_ty = tokens[tok];
  auto tok_pos = lx.token_pos[tok];

  switch (ast_ty) {
  case (AST_UNARY):
  
    if (prec.findUnary(tok_ty) == -1)
      err += error(is, "Expected a unary operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_BINOP):

    if (prec.findBinary(tok_ty) == -1)
      err += error(is, "Expected a binary operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_REDUCE_OP):

    if (prec.findBinary(tok_ty) == -1 && tok_ty != TOK_IDENT)
      err += error(is, "Expected an identifier or a binary operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_ASSIGN):

    if (tok_ty != '=')
      err += error(is, "Expected assignment operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_FN_CALL):
  case (AST_VAR):
  case (AST_ARR_INDEX):

    if (tok_ty != TOK_IDENT)
      err += error(is, "Expected an identifier.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_LIT_INT):
    
    if (tok_ty != TOK_INT_LIT)
      err += error(is, "Expected an integer literal.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_LIT_REAL):
    
    if (tok_ty != TOK_REAL_LIT)
      err += error(is, "Expected a real literal.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (AST_LIT_STRING):
    
    if (tok_ty != TOK_STRING_LIT)
      err += error(is, "Expected a string literal.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;
    
  case (AST_IF):
  case (AST_IF_COND):
  case (AST_IF_BODY):
  case (AST_ELIF_COND):
  case (AST_ELIF_BODY):
  case (AST_ELSE_BODY):
  case (AST_FOR):
  case (AST_FOREACH):
  case (AST_RANGE):
  case (AST_USE):
  case (AST_REDUCE):
  case (AST_ARR):
  case (AST_EXPR_LIST):
  case (AST_FN_DEF):
  case (AST_FN_ARGS):
  case (AST_BLOCK):
  case (AST_RETURN):
    node = tree.addNode(ast_tok, ast_ty, parent);
    break;

  default:
    err += error(is, "Unknown node", ast_pos);
  }

  // scan to next )
  while(tok<tokens.size() && tokens[tok] != ')' && tokens[tok] != '(') 
  { consume(tokens.size(), tok); }

  return {err, node};
}

//==============================================================================
// Main sext parse function
//==============================================================================
int parse_sext(
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & res)
{
  const auto & tokens = lx.tokens;
  auto ntokens = tokens.size();
  int i=0;
  auto err = 0;

  std::stack<int> q;
  int current = -1, root = -1;

  while (i < ntokens) {

    auto tok = tokens[i];

    if (tok == TOK_EOF)
      break;
    else if (tok == '(') {
      ++i;
      auto [e, node] = parse_sext_node(current, is, lx, prec, res, i);
      err += e;
      if (current == -1) root = node;
      q.push(current);
      current = node;
    }
    else if (tok == ')') {
      if (!q.size()) {
        err += error(is, "Unmatched closing paren, ')'.", lx.token_pos[i]);
      }
      current = q.top();
      q.pop();
      ++i;
    }
    
  }
      
  if (q.size()) {
    err += error(is, "Unmatched opening paren, '('.", lx.token_pos[i]);
  }
      
  return err;
}

} // namespace
