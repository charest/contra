#include "errors.hpp"
#include "identifier.hpp"
#include "graph.hpp"
#include "parser.hpp"
#include "stream.hpp"

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

bool isType(int ty)
{
  switch (ty) {
  case tok_i64:
  case tok_f64:
    return true;
  default:
    return false;
  };
}

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
    printRight(os, sw, ' ', ast_to_string(ty));
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
  const Tokens & toks,
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
    os << space << "(" << ast_to_string(ty);

    switch (ty) {
    case ast_fn_def:
    case ast_fn_call:
    case ast_var:
    case ast_arr_index:
    case ast_lit_real:
    case ast_lit_int:
    case ast_lit_string:
    case ast_arr: {
      auto id = lex.findIdentifier(tid);
      os << " " << lex.getIdentifierString(id);
      break;
    }
    
    case ast_assign:
    case ast_unary:
    case ast_binop: {
      os << " " << toks.findInAll(tok);
      break;
    }

    case ast_reduce_op: {
      if (tok == tok_ident) {
        auto id = lex.findIdentifier(tid);
        os << " " << lex.getIdentifierString(id);
      }
      else {
        os << toks.findInAll(tok);
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
  const Tokens & toka,
  const lexed_t & lxa,
  const parse_tree_t & tra,
  const Tokens & tokb,
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
    std::cerr << "{" << na << ", " << ast_to_string(ast_tya);
    std::cerr << "} vs {";
    std::cerr << nb << ", " << ast_to_string(ast_tyb) << "}";
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

  case (ast_unary):
  case (ast_binop):
  case (ast_assign):
    if (tya != tyb) {
      std::cerr << "Operators don't match. ";
      std::cerr << "{" << na << ", " << toka.findInAll(tya);
      std::cerr << "} vs {";
      std::cerr << nb << ", " << tokb.findInAll(tyb) << "}";
      std::cerr << std::endl;
      return false;
    }
    break;
  
  case (ast_fn_call):
  case (ast_var):
  case (ast_arr_index):
  case (ast_lit_int):
  case (ast_lit_real):
  case (ast_lit_string):
  
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
  
  case (ast_reduce_op):

    if (tya != tyb || stra != strb) {
      std::cerr << "Operators or identifiers don't match. ";
      std::cerr << "{" << na << ", " << toka.findInAll(tya) << ", " << stra;
      std::cerr << "} vs {";
      std::cerr << nb << ", " << tokb.findInAll(tyb) << ", " << strb << "}";
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
  const Tokens & toka,
  const lexed_t & lxa,
  const parse_tree_t & tra,
  const graph_t & gra,
  stream_t & isb,
  const Tokens & tokb,
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
    auto new_node = tree.addNode(tok, ast_unary, parent);
    ++tok;
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
    auto binop = tree.addNode(tok, ast_binop, parent);
    tree.setParent(lhs.node, binop); // repoint the left
    
    ++tok; // eat binop
    
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
  if (tokens[tok] == tok_comma) {
    // create new root node
    auto expr_list = tree.addNode(tok, ast_expr_list, parent);
    // repoint the old root
    tree.setParent(lhs, expr_list);
    // add to the list
    while (tokens[tok] == tok_comma) {
      ++tok; // eat ,
      auto ret = parse_binop_expr(expr_list, is, lx, prec, tree, tok, 0);
      err += ret.err;
    }
    // set new lhs
    lhs = expr_list;
  }
  else if (tokens[tok] == tok_colon) {
    // create new root node
    auto range_expr = tree.addNode(tok, ast_range, parent);
    // repoint the old root
    tree.setParent(lhs, range_expr);
    // add to the list
    int num_exprs = 1;
    while (tokens[tok] == tok_colon) {
      ++tok; // eat :
      auto res = parse_binop_expr(range_expr, is, lx, prec, tree, tok, 0);
      err += res.err;
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
  
  if (tokens[tok] == tok_asgmt) {
    // create a new root node
    auto assign_expr = tree.addNode(tok, ast_assign, parent);
    ++tok; // eat =
    // repoint the old root
    tree.setParent(lhs, assign_expr);
    // parse the rhs
    auto res = parse_expr(assign_expr, is, lx, prec, tree, tok);
    err += res.err;
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
  parse_tree_t & tree,
  int & tok)
{
  // add token
  return { 0, tree.addNode(tok++, Ty, parent) };
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
  auto node = tree.addNode(tok, ast_return, parent);

  // eat return
  ++tok;
  
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
  ++tok;
  // add expression in parens, parent passes through
  auto res = parse_expr(parent, is, lx, prec, tree, tok);
  // eat )
  if (lx.tokens[tok] != tok_rparens) {
    res += error(is, "Expected ')' after expression", lx.token_pos[tok]);
  }
  ++tok;
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

  ++tok; // eat identifier.

  auto & tokens = lx.tokens;
  
  //----------------------------------------------------------------------------
  // Call.
  if (tokens[tok] == tok_lparens) {
    auto node = tree.addNode(first_tok, ast_fn_call, parent);
    parse_res_t ret{0, node};
    ++tok; // eat (
    if (tokens[tok] != tok_rparens)
      ret += parse_expr(node, is, lx, prec, tree, tok);
    if (tokens[tok] != tok_rparens)
      ret += error(is, "Expected ')'.", lx.token_pos[tok]);
    ++tok; // Eat the ')'.
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
    if (isType(tok_ty)) 
    {
      ident_tok = tok;
      type_tok = first_tok;
      has_type = true;
      if (tokens[tok] != tok_ident)
        ret += error(is, "Expected identifier.", lx.token_pos[tok]);
      ++tok;  // eat the identifier
    }
    else if (tok_ty != tok_ident) 
      ret += error(is, "Expected identifier or type.", lx.token_pos[first_tok]);

    //----------------------------------
    // Array
    if (tokens[tok] == tok_lbrack) {
      ret.node = tree.addNode(ident_tok, ast_arr_index, parent);
      ++tok; // eat [
      ret += parse_expr(ret.node, is, lx, prec, tree, tok);
      if (tokens[tok] != tok_rbrack)
        ret += error(is, "Expected ']'  in array access/declaration.", lx.token_pos[tok]);
      ++tok; // eat ]
    }
    
    //----------------------------------
    // scalar
    else {
      ret.node = tree.addNode(ident_tok, ast_var, parent);
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
    if (lx.tokens[tok] == tok_sep) ++tok;
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
  
  auto if_node = tree.addNode(tok, ast_if, parent);
  parse_res_t ret{0, if_node};

  // TODO if, elseif, else is repeated
  // TODO drop semicolons in lexer

  //---------------------------------------------------------------------------
  // If
  {

    ++tok; // eat the if.

    // 1 - condition.
    auto cond_node = tree.addNode(tok, ast_if_cond, if_node);
    ret += parse_expr(cond_node, is, lx, prec, tree, tok);
    
    // 2 - body.
    auto body_node = tree.addNode(tok, ast_if_body, if_node);
      
    //------------------------------------
    // Multi-liner TODO LIFT OUT BASIC BLOCK
    if (tokens[tok] == tok_lbrace) {
      ++tok; // eat {
      ret.err += parse_until(body_node, is, lx, prec, tree, tok, tok_rbrace);
      ++tok; // eat }
    }
    //------------------------------------
    // One-liner
    else {
      ret += parse_expr(body_node, is, lx, prec, tree, tok);
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (tokens[tok] == tok_elif) {
  
    ++tok; // eat elif

    // 1 - condition.
    auto cond_node = tree.addNode(tok, ast_elif_cond, if_node);
    ret += parse_expr(cond_node, is, lx, prec, tree, tok);

    // 2 - body
    auto body_node = tree.addNode(tok, ast_elif_body, if_node);
  
    //------------------------------------
    // Multi-liner
    if (tokens[tok] == tok_lbrace) {
      ++tok; // eat {
      ret.err += parse_until(body_node, is, lx, prec, tree, tok, tok_rbrace);
      ++tok; // eat }
    }
    //------------------------------------
    // One-liner
    else {
      ret += parse_expr(body_node, is, lx, prec, tree, tok);
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (tokens[tok] == tok_else) {

    ++tok; // eat else
    
    auto body_node = tree.addNode(tok, ast_else_body, if_node);
    
    //------------------------------------
    // Multi-liner
    if (tokens[tok] == tok_lbrace) {
      ++tok; // eat {
      ret.err += parse_until(body_node, is, lx, prec, tree, tok, tok_rbrace);
      ++tok; // eat }
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

  int ast_for_type = (tokens[tok] == tok_foreach) ? ast_foreach : ast_for;
 
  // Top for node
  auto for_node = tree.addNode(tok, ast_for_type, parent);
  parse_res_t ret{0, for_node};

  ++tok; // eat the for.

  if (tokens[tok] != tok_ident)
    ret += error(is, "Expected identifier after 'for'", lx.token_pos[tok]);

  // variable node
  tree.addNode(tok, ast_var, for_node);
  
  ++tok; // eat identifier.

  if (tokens[tok] != tok_asgmt)
    ret += error(is, "Expected '=' after 'for'", lx.token_pos[tok]);
  
  ++tok; // eat =
  
  // range node
  ret += parse_expr(for_node, is, lx, prec, tree, tok);

  // body node
  auto body_node = tree.addNode(tok, ast_for_body, for_node);

  //------------------------------------
  // Multi-liner
  if (tokens[tok] == tok_lbrace) {
    ++tok; // eat {
    ret.err += parse_until(body_node, is, lx, prec, tree, tok, tok_rbrace);
    ++tok; // eat }
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

  auto arr_node = tree.addNode(tok, ast_arr, parent);
  parse_res_t ret{0, arr_node};

  tok++; // eat [.

  ret += parse_expr(arr_node, is, lx, prec, tree, tok);
    
  if (tokens[tok] == tok_sep) {
    ++tok; // eat ;
    ret += parse_expr(arr_node, is, lx, prec, tree, tok);
  }

  if (tokens[tok] != tok_rbrack) {
    ret += error(is, "Expected ']'", lx.token_pos[tok]);
  }
 
  // eat ]
  ++tok;

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
  
  auto node = tree.addNode(tok, ast_reduce, parent);

  ++tok;  // eat the reduce
    
  while (tokens[tok] != tok_colon) {
    if (tokens[tok] != tok_ident)
      err += error(is, "Expected an identifier after keyword 'reduce'.", lx.token_pos[tok]);
    tree.addNode(tok, ast_var, node);  
    ++tok; // eat identifier.
    if (tokens[tok] == tok_comma) ++tok; // eat ,
  }

  if (tokens[tok] != tok_colon)
    err += error(is, "Expected ':'.", lx.token_pos[tok]);
  ++tok; // eat ":".

  auto ty = tokens[tok];
  if ( (prec.findBinary(ty) == -1) && (ty != tok_ident) )
    err += error(is, "Expected identifier or operator after ':'.", lx.token_pos[tok]);

  tree.addNode(tok, ast_reduce_op, node);

  ++tok; // eat identifier

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

  auto node = tree.addNode(tok, ast_use, parent);
    
  ++tok;  // eat the use
  
  while (tokens[tok] != tok_colon) {
    if (tokens[tok] != tok_ident)
      err += error(is, "Expected an identifier after keyword 'use'.", lx.token_pos[tok]);
    tree.addNode(tok, ast_var, node);  
    ++tok; // eat identifier.
    if (tokens[tok] == tok_comma) ++tok; // eat ,
  }

  if (tokens[tok] != tok_colon)
    err += error(is, "Expected ':'.", lx.token_pos[tok]);
  ++tok; // eat ":".

  if (tokens[tok] != tok_ident)
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
  case tok_ident:
  case tok_i64:
  case tok_f64:
    return parse_identifier_expr(parent, is, lx, prec, tree, tok);
  case tok_real_lit:
    return parse_simple_expr<ast_lit_real>(parent, tree, tok);
  case tok_int_lit:
    return parse_simple_expr<ast_lit_int>(parent, tree, tok);
  case tok_lparens:
    return parse_parens_expr(parent, is, lx, prec, tree, tok);
  case tok_lbrack:
    return parse_array_expr(parent, is, lx, prec, tree, tok);
  case tok_if:
    return parse_if_expr(parent, is, lx, prec, tree, tok);
  case tok_for:
  case tok_foreach:
    return parse_for_expr(parent, is, lx, prec, tree, tok);
  case tok_use:
    return parse_part_expr(parent, is, lx, prec, tree, tok);
  case tok_reduce:
    return parse_reduce_expr(parent, is, lx, prec, tree, tok);
  case tok_string_lit:
    return parse_simple_expr<ast_lit_string>(parent, tree, tok);
  case tok_break:
    return parse_simple_expr<ast_break>(parent, tree, tok);
  case tok_return:
    return parse_return(parent, is, lx, prec, tree, tok);
  }
      
  // return parent and hope for recovery
  auto old_tok = tok;
  tok++;
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
      if (tokens[tok] == tok_comma) ++tok; // eat comma
    }

    if (tokens[tok] != tok_ident)
      err += error(is, "Expected function name specification.", lx.token_pos[tok]);
    
    // create the node
    proto = tree.addNode(parent, ast_fn_def, tok);
    
    // add final type and set node
    auto fun_ty = tree.installType( ReturnTypes );  
    tree.setType(proto, fun_ty);

    ++tok; // eat identifier
  }
  // no specified type
  else {
    proto = tree.addNode(parent, ast_fn_def, first_tok);
  }

  
  if (tokens[tok] != tok_lparens)
    err += error(is, "Expected '(' in prototype", lx.token_pos[tok]);

  ++tok; // eat "("

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  while ((tokens[tok]!=tok_rparens) && (tok<ntok)) {

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
    
    if (tokens[tok] == tok_lbrack) {
      IsArray = true;
      ++tok; // eat the '['.
      if (tokens[tok] != tok_rbrack)
        err += error(is, "Expected ']'", lx.token_pos[tok]);
      ++tok; // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (tokens[tok] == tok_comma) ++tok; // eat ','
  }

  if (CurTok_ != tok_rparens)
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
  
  bool IsTask = (tokens[tok] == tok_task);
  auto ast_ty = IsTask ? ast_tsk_def : ast_fn_def;

  ++tok; // eat 'function' / 'task'

  //---------------------------------------------------------------------------
  // Return types
  
  std::vector<int> ReturnTypes;
  int fn_node = -1;

  // know it has specified type
  if (isType(tokens[tok])) {

    while (tokens[tok] != tok_ident) {
      auto tok_ty = tokens[tok];
      if (!isType(tok_ty))
        err += error(is, "Expected type.", lx.token_pos[tok]);
      auto ty = tree.installType(tok_ty);
      ReturnTypes.emplace_back(ty);
      ++tok; // eat type
      if (tokens[tok] == tok_comma) ++tok; // eat comma
    }

    if (tokens[tok] != tok_ident)
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

  ++tok; // eat identifier
  
  if (tokens[tok] != tok_lparens)
    err += error(is, "Expected '(' in prototype", lx.token_pos[tok]);

  ++tok; // eat "("
  
  //---------------------------------------------------------------------------
  // Arguments

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  auto args_node = tree.addNode(tok, ast_fn_args, fn_node);

  while (tokens[tok] != tok_rparens) {

    bool IsArray = false;
    auto type_tok = tok;

    // token must be a type
    if (!isType(tokens[tok]))
      err += error(is, "Type expected.", lx.token_pos[tok]);

    ++tok; // eat type
  
    if (tokens[tok] != tok_ident)
      err += error(is, "Mising type or variable name in prototype.", lx.token_pos[tok]);
   
    auto arg = tree.addNode(tok, ast_var, args_node);
    auto arg_ty = tree.installType(type_tok);
    tree.setType(arg, arg_ty, type_tok);
    
    ++tok; // eat identifier
    
    if (tokens[tok] == tok_lbrack) {
      IsArray = true;
      ++tok; // eat the '['.
      if (tokens[tok] != tok_rbrack)
        err += error(is, "Expected ']'", lx.token_pos[tok]);
      ++tok; // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (tokens[tok] == tok_comma) ++tok; // eat ','
  }

  if (tokens[tok] != tok_rparens)
    err += error(is, "Expected ')' in prototype", lx.token_pos[tok]);

  // success.
  ++tok; // eat ')'.

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
  if (tokens[tok] == tok_lbrace) {
    auto body_node = tree.addNode(tok, ast_fn_body, fn_node);
    ++tok; // eat {
    parse_until(body_node, is, lx, prec, tree, tok, tok_rbrace);
    if (tokens[tok] != tok_rbrace)
      err += error(is, "Expected '}'.", lx.token_pos[tok] );
    ++tok; // eat }
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
    case tok_eof:
      goto exit_loop; // don't hate
    case tok_sep:
      ++i;
      break;
    case tok_task:
    case tok_function:
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
  ++tok;
  auto tok_ty = tokens[tok];
  auto tok_pos = lx.token_pos[tok];

  switch (ast_ty) {
  case (ast_unary):
  
    if (prec.findUnary(tok_ty) == -1)
      err += error(is, "Expected a unary operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_binop):

    if (prec.findBinary(tok_ty) == -1)
      err += error(is, "Expected a binary operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_reduce_op):

    if (prec.findBinary(tok_ty) == -1 && tok_ty != tok_ident)
      err += error(is, "Expected an identifier or a binary operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_assign):

    if (tok_ty != tok_asgmt)
      err += error(is, "Expected assignment operator.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_fn_call):
  case (ast_var):
  case (ast_arr_index):

    if (tok_ty != tok_ident)
      err += error(is, "Expected an identifier.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_lit_int):
    
    if (tok_ty != tok_int_lit)
      err += error(is, "Expected an integer literal.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_lit_real):
    
    if (tok_ty != tok_real_lit)
      err += error(is, "Expected a real literal.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;

  case (ast_lit_string):
    
    if (tok_ty != tok_string_lit)
      err += error(is, "Expected a string literal.", tok_pos);
    node = tree.addNode(tok, ast_ty, parent);
    break;
    
  case (ast_if):
  case (ast_if_cond):
  case (ast_if_body):
  case (ast_elif_cond):
  case (ast_elif_body):
  case (ast_else_body):
  case (ast_for):
  case (ast_foreach):
  case (ast_for_body):
  case (ast_range):
  case (ast_use):
  case (ast_reduce):
  case (ast_arr):
  case (ast_expr_list):
  case (ast_fn_def):
  case (ast_fn_args):
  case (ast_fn_body):
  case (ast_return):
    node = tree.addNode(ast_tok, ast_ty, parent);
    break;

  default:
    err += error(is, "Unknown node", ast_pos);
  }

  // scan to next )
  while(tok<tokens.size() && tokens[tok] != tok_rparens && tokens[tok] != tok_lparens) 
  { ++tok; }

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

    if (tok == tok_eof)
      break;
    else if (tok == tok_lparens) {
      ++i;
      auto [e, node] = parse_sext_node(current, is, lx, prec, res, i);
      err += e;
      if (current == -1) root = node;
      q.push(current);
      current = node;
    }
    else if (tok == tok_rparens) {
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

//==============================================================================
// break
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseBreakExpr() {
  auto Result = std::make_unique<BreakStmtAST>(getIdentifierLoc());
  getNextToken(); // consume the break
  return std::move(Result);
}

  
//==============================================================================
// numberexpr ::= number
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseIntegerExpr() {
  auto Result = std::make_unique<ValueExprAST>(
      getIdentifierLoc(),
      getIdentifierStr(),
      ValueExprAST::ValueType::Int);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// numberexpr ::= number
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseRealExpr() {
  auto Result = std::make_unique<ValueExprAST>(
      getIdentifierLoc(),
      getIdentifierStr(),
      ValueExprAST::ValueType::Real);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// stringexpr ::= string
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseStringExpr() {
  auto Result = std::make_unique<ValueExprAST>(
      getIdentifierLoc(),
      getIdentifierStr(),
      ValueExprAST::ValueType::String);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseParenExpr() {
  auto BeginLoc = getCurLoc();
  getNextToken(); // eat (.
  auto V = parseExpression();

  if (CurTok_ != tok_rparens) {
    THROW_SYNTAX_ERROR(
        "Expected ')' after expression", 
        getLocationRange(BeginLoc) );
  }
  getNextToken(); // eat ).
  return V;
}

//==============================================================================
// identifierexpr
//   ::= 
//   ::= identifier '(' expression* ')'
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseIdentifierExpr() {
  
  auto BeginLoc = getCurLoc();
  auto Id = getIdentifier();

  getNextToken(); // eat identifier.
  
  //----------------------------------------------------------------------------
  // Call.
  if (CurTok_ == tok_lparens) {
    auto ArgsBeginLoc = getCurLoc();
    getNextToken(); // eat (
    std::unique_ptr<NodeAST> Args;
    if (CurTok_ != tok_rparens) Args = parseExpression();
    if (CurTok_ != tok_rparens)
      THROW_NAME_ERROR("Expected ')'.", getLocationRange(ArgsBeginLoc));
    getNextToken(); // Eat the ')'.
    return std::make_unique<CallExprAST>(
        getLocationRange(BeginLoc),
        Id,
        std::move(Args));
  }

  //----------------------------------------------------------------------------
  // Variable reference
  else {
  
    // Has a type, so we know its a decl
    std::unique_ptr<Identifier> VarTypeId;
    if (CurTok_ == tok_ident && isType(Id.getName())) {
      VarTypeId = std::make_unique<Identifier>(Id);
      Id = getIdentifier();
      getNextToken();  // eat the type
    }

    
    //----------------------------------
    // Array
    if (CurTok_ == '[') {
      auto ArrayLoc = getCurLoc();
      getNextToken(); // eat [
      auto IndexExpr = parseExpression();
      if (CurTok_ != tok_rbrack)
        THROW_SYNTAX_ERROR(
            "Expected ']'  in array access/declaration.",
            getLocationRange(ArrayLoc));
      getNextToken(); // eat ]
      return std::make_unique<ArrayAccessExprAST>(
          getLocationRange(BeginLoc),
          Id,
          std::move(IndexExpr),
          std::move(VarTypeId));
    }
    
    //----------------------------------
    // scalar
    else {
      return std::make_unique<VarAccessExprAST>(
          getLocationRange(BeginLoc),
          Id,
          std::move(VarTypeId));
    }

  } // variable reference

}

//==============================================================================
// ifexpr ::= 'if' expression 'then' expression 'else' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseIfExpr() {

  IfStmtAST::ConditionList Conds;
  ASTBlockList BBlocks;
  
  //---------------------------------------------------------------------------
  // If
  {

    auto IfLoc = getIdentifierLoc();
    getNextToken(); // eat the if.

    // condition.
    auto Cond = parseExpression();
    Conds.emplace_back( IfLoc, std::move(Cond) );
      
    // make a new block
    auto Then = createBlock(BBlocks);
  
    //------------------------------------
    // Multi-liner
    if (CurTok_ == tok_lbrace) {
      getNextToken(); // eat {
      while (CurTok_ != tok_rbrace) {
        auto E = parseExpression();
        Then->emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      auto E = parseExpression();
      Then->emplace_back( std::move(E) );
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (CurTok_ == tok_elif) {
  
    auto ElifLoc = getIdentifierLoc();
    getNextToken(); // eat elif

    // condition.
    auto Cond = parseExpression();
    Conds.emplace_back( ElifLoc, std::move(Cond) );
  
    // make a new block
    auto Then = createBlock(BBlocks);

    //------------------------------------
    // Multi-liner
    if (CurTok_ == tok_lbrace) {
      getNextToken(); // eat {
      while (CurTok_ != tok_rbrace) {
        auto E = parseExpression();
        Then->emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      auto E = parseExpression();
      Then->emplace_back( std::move(E) );
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (CurTok_ == tok_else) {

    getNextToken(); // eat else
    
    // make a new block
    auto Else = createBlock(BBlocks);

    //------------------------------------
    // Multi-liner
    if (CurTok_ == tok_lbrace) {
      getNextToken(); // eat {
      while (CurTok_ != tok_rbrace) {
        auto E = parseExpression();
        Else->emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      auto E = parseExpression();
      Else->emplace_back( std::move(E) );
    }

  }
  
  //---------------------------------------------------------------------------
  // Construct If Else Then tree

  return IfStmtAST::makeNested( Conds, BBlocks );
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseForExpr() {
  auto BeginLoc = getCurLoc();
  auto ForLoc = getIdentifierLoc();
  
  bool IsForEach = CurTok_ == tok_foreach;

  getNextToken(); // eat the for.

  if (CurTok_ != tok_ident)
    THROW_SYNTAX_ERROR("Expected identifier after 'for'", getIdentifierLoc());
  std::string IdName = getIdentifierStr();
  auto IdentLoc = getIdentifierLoc();
  getNextToken(); // eat identifier.

  if (CurTok_ != tok_asgmt)
    THROW_SYNTAX_ERROR(
        "Expected '=' after 'for'",
        getLocationRange(BeginLoc));
  getNextToken(); // eat =

  auto Start = parseExpression();

  // add statements
  ASTBlock Body;

  //------------------------------------
  // Multi-liner
  if (CurTok_ == tok_lbrace) {
    getNextToken(); // eat {
    while (CurTok_ != tok_rbrace) {
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }
    getNextToken(); // eat }
  }
  //------------------------------------
  // One-liner
  else {
    auto E = parseExpression();
    Body.emplace_back( std::move(E) );
  }
  
  // make a for loop
  auto Id = Identifier{IdName, IdentLoc};
  std::unique_ptr<NodeAST> F;
  if (IsForEach)
    F = std::make_unique<ForeachStmtAST>(
        ForLoc,
        Id,
        std::move(Start),
      std::move(Body));
  else
    F = std::make_unique<ForStmtAST>(
        ForLoc,
        Id,
        std::move(Start),
        std::move(Body));


  return F;
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
std::unique_ptr<NodeAST> Parser::parsePrimary() {
 
  switch (CurTok_) {
  case tok_ident:
    return parseIdentifierExpr();
  case tok_real_lit:
    return parseRealExpr();
  case tok_int_lit:
    return parseIntegerExpr();
  case tok_lparens:
    return parseParenExpr();
  case '[':
    return parseArrayExpr();
  case tok_if:
    return parseIfExpr();
  case tok_for:
  case tok_foreach:
    return parseForExpr();
  case tok_use:
    return parsePartitionExpr();
  case tok_reduce:
    return parseReductionExpr();
  case tok_string_lit:
    return parseStringExpr();
  case tok_break:
    return parseBreakExpr();
  default:
    THROW_SYNTAX_ERROR("Unknown token '" <<  Tokens::getName(CurTok_)
        << "' when expecting an expression", getIdentifierLoc());
  }
}

//==============================================================================
// binoprhs
//   ::= ('+' primary)*
//==============================================================================
std::unique_ptr<NodeAST>
Parser::parseBinOpRHS(int ExprPrec, std::unique_ptr<NodeAST> LHS)
{
  
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = getTokPrecedence();
    
    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok_;
    auto BinLoc = getIdentifierLoc();
    getNextToken(); // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = parseUnary();

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = getTokPrecedence();

    if (TokPrec < NextPrec) {
      RHS = parseBinOpRHS(TokPrec + 1, std::move(RHS));
    }

    // Merge LHS/RHS.
    LHS = std::make_unique<BinaryExprAST>(BinLoc, BinOp, std::move(LHS),
        std::move(RHS));
  }

  return nullptr;
}

//==============================================================================
// expression
//   ::= primary binoprhs
//
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseExpression() {
  auto BeginLoc = getCurLoc();
  
  std::unique_ptr<NodeAST> LHS = parseUnary();
  LHS = parseBinOpRHS(0, std::move(LHS));

  if (CurTok_ == tok_comma) {
    ASTBlock Exprs;
    Exprs.emplace_back( std::move(LHS) );
    while (CurTok_ == tok_comma) {
      getNextToken(); // eat ,
      std::unique_ptr<NodeAST> LHS = parseUnary();
      LHS = parseBinOpRHS(0, std::move(LHS));
      Exprs.emplace_back( std::move(LHS) );
    }
    LHS = std::make_unique<ExprListAST>(
        getLocationRange(BeginLoc),
        std::move(Exprs));
  }
  else if (CurTok_ == tok_colon) {
    ASTBlock Exprs;
    Exprs.emplace_back( std::move(LHS) );
    while (CurTok_ == tok_colon) {
      getNextToken(); // eat :
      std::unique_ptr<NodeAST> LHS = parseUnary();
      LHS = parseBinOpRHS(0, std::move(LHS));
      Exprs.emplace_back( std::move(LHS) );
    }
    if (Exprs.size() > 3 || Exprs.size() < 2)
      THROW_SYNTAX_ERROR(
          "Only 'begin':'end':['step'] specification supported for ranges." ,
          getLocationRange(BeginLoc));
    LHS = std::make_unique<RangeExprAST>(
        getLocationRange(BeginLoc),
        std::move(Exprs));
  }

  if (CurTok_ == tok_asgmt) {
    getNextToken(); // eat =
    auto RHS = parseExpression();
    LHS = std::make_unique<AssignStmtAST>(
        getLocationRange(BeginLoc),
        std::move(LHS),
        std::move(RHS));
  }

  return LHS;
}

//==============================================================================
// toplevelexpr ::= expression
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseTopLevelExpr() {

  auto FnLoc = getIdentifierLoc();
  auto E = parseExpression();
  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>( Identifier{"__anon_expr", FnLoc} );
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
}

//==============================================================================
// unary
//   ::= primary
//   ::= '!' unary
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseUnary() {

  // If the current token is not an operator, it must be a primary expr.
  if (!isTokOperator() || CurTok_ == tok_lparens || CurTok_ == tok_comma) {
    auto P = parsePrimary();
    return P;
  }

  // If this is a unary operator, read it.
  int Opc = CurTok_;
  getNextToken();
  auto Operand = parseUnary();
  return std::make_unique<UnaryExprAST>(
      getIdentifierLoc(),
      Opc,
      std::move(Operand));
}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parsePartitionExpr() {

  auto BeginLoc = getCurLoc();
  getNextToken();  // eat the use
    
  std::vector<Identifier> RangeIds;

  while (CurTok_ != tok_colon) {
    auto RangeLoc = getIdentifierLoc();
    if (CurTok_ != tok_ident)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'use'.", RangeLoc);
    RangeIds.emplace_back( getIdentifierStr(), RangeLoc );
    getNextToken(); // eat identifier.
    if (CurTok_ == tok_comma) getNextToken(); // eat ,
  }

  auto ColonLoc = getCurLoc();
  if (CurTok_ != tok_colon)
    THROW_SYNTAX_ERROR(
        "Expected ':'.",
        getIdentifierLoc());
  getNextToken(); // eat ":".

  if (CurTok_ != tok_ident)
    THROW_SYNTAX_ERROR(
        "Expected identifier after ':'.",
        getLocationRange(ColonLoc));
  auto PartExpr = parseExpression(); 

  return std::make_unique<PartitionStmtAST>(
      getLocationRange(BeginLoc),
      RangeIds,
      std::move(PartExpr));
}

//==============================================================================
// reduction
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseReductionExpr() {

  auto BeginLoc = getCurLoc();
  getNextToken();  // eat the reduce
    
  std::vector<Identifier> VarIds;

  while (CurTok_ != tok_colon) {
    auto VarLoc = getIdentifierLoc();
    if (CurTok_ != tok_ident)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'reduce'.", VarLoc);
    VarIds.emplace_back( getIdentifierStr(), VarLoc );
    getNextToken(); // eat identifier.
    if (CurTok_ == tok_comma) getNextToken(); // eat ,
  }

  if (CurTok_ != tok_colon)
    THROW_SYNTAX_ERROR(
        "Expected ':'.",
        getIdentifierLoc());
  getNextToken(); // eat ":".

  if (!isTokOperator() && (CurTok_ != tok_ident))
    THROW_SYNTAX_ERROR(
        "Expected identifier or operator after ':'.",
        getIdentifierLoc());

  auto OperatorLoc = getIdentifierLoc();
  std::unique_ptr<NodeAST> Expr;

  if (isTokOperator()) {
    Expr = std::make_unique<ReductionStmtAST>(
        getLocationRange(BeginLoc),
        VarIds,
        CurTok_,
        Tokens::getName(CurTok_),
        OperatorLoc);
  }
  else {
    auto OperatorStr = getIdentifierStr();
    Expr = std::make_unique<ReductionStmtAST>(
        getLocationRange(BeginLoc),
        VarIds,
        OperatorStr,
        OperatorLoc);
  }
  getNextToken(); // eat identifier

  return Expr;

}

//==============================================================================
// Array expression parser
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseArrayExpr()
{

  auto BeginLoc = getCurLoc();
  getNextToken(); // eat [.

  std::unique_ptr<NodeAST> SizeExpr;
  auto ValExprs = parseExpression();
    
  if (CurTok_ == tok_sep) {
    getNextToken(); // eat ;
    SizeExpr = parseExpression();
  }

  if (CurTok_ != tok_rbrack)
    THROW_SYNTAX_ERROR(
        "Expected ']'",
        LocationRange(BeginLoc, getCurLoc()) );

 
  // eat ]
  getNextToken();

  return std::make_unique<ArrayExprAST>(
      LocationRange(BeginLoc, getCurLoc()),
      std::move(ValExprs),
      std::move(SizeExpr));
}

//==============================================================================
// Toplevel function parser
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseFunction() {

  bool IsTask = (CurTok_ == tok_task);

  getNextToken(); // eat 'function' / 'task'
  auto Proto = parsePrototype();

  ASTBlock Body;
  std::unique_ptr<NodeAST> Return;

  //------------------------------------
  // Multi-liner
  if (CurTok_ == tok_lbrace) {
    getNextToken(); // eat {
    while (CurTok_ != tok_rbrace) {
      if (CurTok_ == tok_return) {
        getNextToken(); // eat return
        Return = parseExpression();
        break;
      }
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }
    if (CurTok_ != tok_rbrace)
      THROW_SYNTAX_ERROR(
          "Only one return statement allowed for a function.",
          getIdentifierLoc() );
    getNextToken(); // eat }
  }
  //------------------------------------
  // One-liner
  else {
    if (CurTok_ == tok_return) {
      getNextToken(); // eat return
      Return = parseExpression();
    }
    else {
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
    }
  }

  
  if (IsTask) {
    return std::make_unique<TaskAST>(
        std::move(Proto),
        std::move(Body),
        std::move(Return));
  }
  else
    return std::make_unique<FunctionAST>(
        std::move(Proto),
        std::move(Body),
        std::move(Return));
}

//==============================================================================
// prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parsePrototype() {

  std::string FnName;
  auto BeginLoc = getCurLoc();

  auto FnLoc = getIdentifierLoc();

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok_) {
  default:
    THROW_SYNTAX_ERROR(
        "Expected function name in prototype", 
        FnLoc);
  case tok_ident:
    FnName = getIdentifierStr();
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR(
          "Expected unary operator",
          getIdentifierLoc());
    FnName = "unary";
    FnName += (char)CurTok_;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR(
          "Expected binrary operator",
          getIdentifierLoc());
    FnName = "binary";
    FnName += (char)CurTok_;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok_ == tok_int_lit) {
      auto NumVal = std::stoi(getIdentifierStr());
      if (NumVal < 1 || NumVal > 100)
        THROW_SYNTAX_ERROR(
            "Invalid precedence of '" << NumVal
            << "' must be between 1 and 100",
            getIdentifierLoc());
      BinaryPrecedence = NumVal;
      getNextToken();
    }
    else {
      THROW_SYNTAX_ERROR(
          "Precedence must be an integer number",
          getIdentifierLoc());
    }
    break;
  }
  
  std::vector<Identifier> ReturnTypes;

  // know it has specified arguments
  if (CurTok_ == tok_comma || CurTok_ == tok_ident) {
    ReturnTypes.emplace_back(FnName, FnLoc);
    while (CurTok_ == tok_comma) {
      getNextToken(); // eat ,
      if (CurTok_ != tok_ident)
        THROW_SYNTAX_ERROR(
            "Expected identifier in return type specification.",
            getLocationRange(BeginLoc));
      ReturnTypes.emplace_back(getIdentifierStr(), getIdentifierLoc());
      getNextToken(); // eat identifier
    }
      
    if (CurTok_ != tok_ident) {
        THROW_SYNTAX_ERROR(
            "Expected function name specification.",
            getLocationRange(BeginLoc));
    }
    FnName = getIdentifierStr();
    FnLoc = getIdentifierLoc();
    getNextToken();
  }

  
  if (CurTok_ != tok_lparens)
    THROW_SYNTAX_ERROR(
        "Expected '(' in prototype",
        getIdentifierLoc());

  getNextToken(); // eat "("

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  while (CurTok_ == tok_ident) {

    bool IsArray = false;

    auto BeginLoc = getCurLoc();

    if (CurTok_ != tok_ident)
      THROW_SYNTAX_ERROR(
          "Identifier expected n prototype for function '" << FnName << "'",
          getLocationRange(BeginLoc));

    auto TypeLoc = getIdentifierLoc();
    auto TypeName = getIdentifierStr();
    ArgTypes.emplace_back( TypeName, TypeLoc );

    getNextToken(); // eat identifier
    
    if (CurTok_ != tok_ident)
      THROW_SYNTAX_ERROR(
          "Mising type or variable name in prototype for function '" 
          << FnName << "'",
          getLocationRange(BeginLoc));
    auto VarName = getIdentifierStr();
    auto VarLoc = getIdentifierLoc();
    
    Args.emplace_back( VarName, VarLoc );
    
    getNextToken(); // eat identifier
    
    if (CurTok_ == tok_lbrack) {
      IsArray = true;
      auto BeginLoc = getCurLoc();
      getNextToken(); // eat the '['.
      if (CurTok_ != tok_rbrack)
        THROW_SYNTAX_ERROR(
            "Expected ']'",
            getLocationRange(BeginLoc));
      getNextToken(); // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (CurTok_ == tok_comma) getNextToken(); // eat ','
  }

  if (CurTok_ != tok_rparens)
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

} // namespace
