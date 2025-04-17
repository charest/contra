#include "errors.hpp"
#include "identifier.hpp"
#include "graph.hpp"
#include "parser.hpp"

#include "utils/string_utils.hpp"

#include <iomanip>
#include <list>
#include <map>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

namespace contra {

//==============================================================================
/// Dump parser results
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
/// Dump parser results
//==============================================================================
void print(
  std::ostream& os,
  const Tokens & toks,
  const lexer_results_t & lex,
  const parse_tree_t & tree,
  const graph_t & graph
)
{
  auto n = tree.size();

  std::stack<std::pair<int,int>> q;
  
  for (auto r : graph.roots) q.push({r,0});
    
  while (q.size()) {
    auto curr = q.top();
    auto i = curr.first;
    auto depth = curr.second;
    auto space = std::string(2*depth, ' ');
    q.pop();

    auto ty = tree.node_ast_type[i];
    auto tid = tree.node_to_token[i];
    auto tok = lex.tokens[tid];
    os << space << ast_to_string(ty) << "(Id = " << i;

    switch (ty) {
    case ast_fn_def:
    case ast_fn_call:
    case ast_access_var:
    case ast_access_arr:
    case ast_value_real:
    case ast_value_int:
    case ast_value_string:
    case ast_arr: {
      auto id = lex.findIdentifier(tid);
      os << ", Ident = \"" << lex.getIdentifierString(id) << "\"";
      break;
    }
    case ast_assign:
    case ast_unary:
    case ast_binop: {
      os << ", Op = \'" << toks.findInAll(tok) << "\'";
      break;
    }}
    
    os << ")" << std::endl;

    auto nc = graph.size(i);
    for (int c=nc; c-->0; ) q.push({graph(i,c), depth+1});
  }
  os << std::endl;

}

//==============================================================================
int parse_primary_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok);

//==============================================================================
// unary
//   ::= primary
//   ::= '!' unary
//==============================================================================
int parse_unary_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  // If the current token is not an operator, it must be a primary expr.
  if ((prec.find_v2(tokens[tok]) == -1) ||
      tokens[tok] == tok_lparens ||
      tokens[tok] == tok_comma) 
  {
    return parse_primary_expr(parent, tokens, prec, tree, tok);
  }
  // If this is a unary operator, read it.
  else {
    auto op_node = tree.addNode(tok, ast_unary, parent);
    ++tok;
    parse_unary_expr(op_node, tokens, prec, tree, tok);
    return op_node;
  }
}

//==============================================================================
// binoprhs
//   ::= ('+' primary)*
//==============================================================================
int parse_binop_expr(
  int parent,
  int lhs,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok,
  int ExprPrec)
{

  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = prec.find_v2(tokens[tok]);
    
    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return lhs;

    // Okay, we know this is a binop.
    auto binop = tree.addNode(tok, ast_binop, parent);
    tree.setParent(lhs, binop); // repoint the left
    
    ++tok; // eat binop
    
    // Parse the unary expression after the binary operator.
    auto rhs = parse_unary_expr(binop, tokens, prec, tree, tok);
  

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = prec.find_v2(tokens[tok]);

    if (TokPrec < NextPrec)
      rhs = parse_binop_expr(binop, rhs, tokens, prec, tree, tok, TokPrec+1);
    
    // Move the binop to the left
    lhs = binop;

  }
  
  return -1; // shouldnt get here
}



//==============================================================================
// expression
//   ::= primary binoprhs
//==============================================================================
int parse_single_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  auto lhs = parse_unary_expr(parent, tokens, prec, tree, tok);
  return parse_binop_expr(parent, lhs, tokens, prec, tree, tok, 0);
}


//==============================================================================
// expression
//   ::= primary binoprhs , primary binoprhs
//   ::= primary binoprhs : primary binoprhs
//   ::= primary binoprhs = primary binoprhs
//==============================================================================
int parse_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  // parse first expr
  auto lhs = parse_single_expr(parent, tokens, prec, tree, tok);
  
  // hit a list
  if (tokens[tok] == tok_comma) {
    // create new root node
    auto list_expr = tree.addNode(tok, ast_expr_list, parent);
    // repoint the old root
    tree.setParent(lhs, list_expr);
    // add to the list
    while (tokens[tok] == tok_comma) {
      ++tok; // eat ,
      parse_single_expr(list_expr, tokens, prec, tree, tok);
    }
    // set new lhs
    lhs = list_expr;
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
      parse_single_expr(range_expr, tokens, prec, tree, tok);
      num_exprs++;
    }
    // validate the number of expressions found
    if (num_exprs > 3 || num_exprs < 2)
      THROW_PARSER_ERROR(
          "Only 'begin':'end':['step'] specification supported for ranges." ,
          tok);
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
    parse_expr(assign_expr, tokens, prec, tree, tok);
    // set new lhs
    lhs = assign_expr;
  }

  return lhs;
}

//==============================================================================
// numberexpr ::= number
// breakexpr
//==============================================================================
// BreakStmtAST - ast_break
// ValueExprAST - ast_value_int, ast_value_real, ast_value_string
template<int Ty>
int parse_simple_expr(
  int parent,
  const std::vector<int> & tokens,
  parse_tree_t & tree,
  int & tok) 
{
  // add token
  return tree.addNode(tok++, Ty, parent);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
int parse_parens_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok) 
{ 
  // eat (
  ++tok;
  // add expression in parens, parent passes through
  auto ret = parse_expr(parent, tokens, prec, tree, tok);
  // eat )
  if (tokens[tok] != ')')
    THROW_PARSER_ERROR("Expected ')' after expression", tok);
  ++tok;
  return ret;
}


//==============================================================================
// identifierexpr
//   ::= 
//   ::= identifier '(' expression* ')'
//==============================================================================
int parse_identifier_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok) 
{
  // store identifier token
  auto ident_tok = tok;

  ++tok; // eat identifier.
  
  //----------------------------------------------------------------------------
  // Call.
  if (tokens[tok] == '(') {
    auto call_node = tree.addNode(ident_tok, ast_fn_call, parent);
    ++tok; // eat (
    if (tokens[tok] != ')') parse_expr(call_node, tokens, prec, tree, tok);
    if (tokens[tok] != ')') THROW_PARSER_ERROR("Expected ')'.", tok);
    ++tok; // Eat the ')'.
    return call_node;
  }

  //----------------------------------------------------------------------------
  // Variable reference
  else {
      
    // Has a type, so we know its a decl
    bool has_type = false;
    int type_tok = -1;
    if (tokens[tok] == tok_identifier) {
      type_tok = tok;
      has_type = true;
      ++tok;  // eat the type
    }

    int var_node;

    //----------------------------------
    // Array
    if (tokens[tok] == '[') {
      var_node = tree.addNode(ident_tok, ast_access_arr, parent);
      ++tok; // eat [
      parse_expr(var_node, tokens, prec, tree, tok);
      if (tokens[tok] != ']')
        THROW_PARSER_ERROR("Expected ']'  in array access/declaration.", tok);
      ++tok; // eat ]
    }
    
    //----------------------------------
    // scalar
    else {
      var_node = tree.addNode(ident_tok, ast_access_var, parent);
    }

    // add type info if any
    if (has_type) tree.setType(var_node, type_tok);

    return var_node;

  } // variable reference

}

//==============================================================================
// ifexpr ::= 'if' expression 'then' expression 'else' expression
//==============================================================================
int parse_if_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok) 
{
  auto if_node = tree.addNode(tok, ast_if, parent);

  // TODO if, elseif, else is repeated
  // TODO drop semicolons in lexer

  //---------------------------------------------------------------------------
  // If
  {

    ++tok; // eat the if.

    // 1 - condition.
    auto cond_node = tree.addNode(tok, ast_if_cond, if_node);
    parse_expr(cond_node, tokens, prec, tree, tok);
    
    // 2 - body.
    auto body_node = tree.addNode(tok, ast_if_body, if_node);
      
    //------------------------------------
    // Multi-liner TODO LIFT OUT BASIC BLOCK
    if (tokens[tok] == '{') {
      ++tok; // eat {
      while (tokens[tok] != '}') {
        parse_expr(body_node, tokens, prec, tree, tok);
        if (tokens[tok] == tok_sep) ++tok;
      }
      ++tok; // eat }
    }
    //------------------------------------
    // One-liner
    else {
      parse_expr(body_node, tokens, prec, tree, tok);
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (tokens[tok] == tok_elif) {
  
    ++tok; // eat elif

    // 1 - condition.
    auto cond_node = tree.addNode(tok, ast_elif_cond, if_node);
    parse_expr(cond_node, tokens, prec, tree, tok);

    // 2 - body
    auto body_node = tree.addNode(tok, ast_elif_body, if_node);
  
    //------------------------------------
    // Multi-liner
    if (tokens[tok] == '{') {
      ++tok; // eat {
      while (tokens[tok] != '}') {
        parse_expr(body_node, tokens, prec, tree, tok);
        if (tokens[tok] == tok_sep) ++tok;
      }
      ++tok; // eat }
    }
    //------------------------------------
    // One-liner
    else {
      parse_expr(body_node, tokens, prec, tree, tok);
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (tokens[tok] == tok_else) {

    ++tok; // eat else
    
    auto body_node = tree.addNode(tok, ast_else_body, if_node);
    
    //------------------------------------
    // Multi-liner
    if (tokens[tok] == '{') {
      ++tok; // eat {
      while (tokens[tok] != '}') {
        parse_expr(body_node, tokens, prec, tree, tok);
        if (tokens[tok] == tok_sep) ++tok;
      }
      ++tok; // eat }
    }
    //------------------------------------
    // One-liner
    else {
      parse_expr(body_node, tokens, prec, tree, tok);
    }

  }

  return if_node;
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
int parse_for_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
  
  int ast_for_type = (tokens[tok] == tok_foreach) ? ast_foreach : ast_for;
  
  auto for_node = tree.addNode(tok, ast_for_type, parent);

  ++tok; // eat the for.

  if (tokens[tok] != tok_identifier)
    THROW_PARSER_ERROR("Expected identifier after 'for'", tok);

  auto ident_tok = tok;
  ++tok; // eat identifier.

  if (tokens[tok] != tok_asgmt)
    THROW_PARSER_ERROR("Expected '=' after 'for'", tok);
  ++tok; // eat =
  
  
  auto start_node = tree.addNode(ident_tok, ast_for_range, for_node);
  parse_expr(start_node, tokens, prec, tree, ident_tok);

  // add statements
  auto body_node = tree.addNode(ident_tok, ast_for_body, for_node);

  //------------------------------------
  // Multi-liner
  if (tokens[tok] == '{') {
    ++tok; // eat {
    while (tokens[tok] != '}') {
      parse_expr(body_node, tokens, prec, tree, tok);
      if (tokens[tok] == tok_sep) ++tok;
    }
    ++tok; // eat }
  }
  //------------------------------------
  // One-liner
  else {
    parse_expr(body_node, tokens, prec, tree, tok);
  }

  return for_node;
  
}

//==============================================================================
// Array expression parser
//==============================================================================
int parse_array_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{

  
  auto arr_node = tree.addNode(tok, ast_arr, parent);

  tok++; // eat [.

  parse_expr(arr_node, tokens, prec, tree, tok);
    
  if (tokens[tok] == ';') {
    ++tok; // eat ;
    parse_expr(arr_node, tokens, prec, tree, tok);
  }

  if (tokens[tok] != ']')
    THROW_PARSER_ERROR("Expected ']'", tok);
 
  // eat ]
  ++tok;

  return arr_node;

}

//==============================================================================
// reduction
//==============================================================================
int parse_reduce_expr(
  int parent,
  const std::vector<int> & tokens,
  parse_tree_t & tree,
  int & tok)
{

  ++tok;  // eat the reduce
    
  std::vector<int> idents; // TODO what do with idents

  while (tokens[tok] != ':') {
    if (tokens[tok] != tok_identifier)
      THROW_PARSER_ERROR("Expected an identifier after keyword 'reduce'.", tok);
    idents.emplace_back(tok);
    ++tok; // eat identifier.
    if (tokens[tok] == ',') ++tok; // eat ,
  }

  if (tokens[tok] != ':')
    THROW_PARSER_ERROR("Expected ':'.", tok);
  ++tok; // eat ":".

  if (tok != tok_identifier)
    THROW_PARSER_ERROR("Expected identifier or operator after ':'.", tok);

  auto ret = tree.addNode(tok, ast_reduce, parent);

  ++tok; // eat identifier

  return ret;

}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
int parse_part_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{


  ++tok;  // eat the use
    
  std::vector<int> idents; // TODO what to do with idents?

  while (tokens[tok] != ':') {
    if (tokens[tok] != tok_identifier)
      THROW_PARSER_ERROR("Expected an identifier after keyword 'use'.", tok);
    idents.emplace_back( tok );
    ++tok; // eat identifier.
    if (tokens[tok] == ',') ++tok; // eat ,
  }

  if (tokens[tok] != ':')
    THROW_PARSER_ERROR("Expected ':'.", tok);
  ++tok; // eat ":".

  if (tokens[tok] != tok_identifier)
    THROW_PARSER_ERROR("Expected identifier after ':'.", tok);

  auto node = tree.addNode(tok, ast_reduce, parent);
  parse_expr(node, tokens, prec, tree, tok);

  return node;
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
int parse_primary_expr(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int & tok)
{
 
  switch (tokens[tok]) {
  case tok_identifier:
    return parse_identifier_expr(parent, tokens, prec, tree, tok);
  case tok_real_literal:
    return parse_simple_expr<ast_value_real>(parent, tokens, tree, tok);
  case tok_int_literal:
    return parse_simple_expr<ast_value_int>(parent, tokens, tree, tok);
  case '(':
    return parse_parens_expr(parent, tokens, prec, tree, tok);
  case '[':
    return parse_array_expr(parent, tokens, prec, tree, tok);
  case tok_if:
    return parse_if_expr(parent, tokens, prec, tree, tok);
  case tok_for:
  case tok_foreach:
    return parse_for_expr(parent, tokens, prec, tree, tok);
  case tok_use:
    return parse_part_expr(parent, tokens, prec, tree, tok);
  case tok_reduce:
    return parse_reduce_expr(parent, tokens, tree, tok);
  case tok_string_literal:
    return parse_simple_expr<ast_value_string>(parent, tokens, tree, tok);
  case tok_break:
    return parse_simple_expr<ast_break>(parent, tokens, tree, tok);
  default:
    THROW_PARSER_ERROR("Unknown token '" <<  Tokens::getName(tokens[tok])
        << "' when expecting an expression", tok);
  }
}

//==============================================================================
// toplevelexpr ::= expression
//==============================================================================
void parse_top_level(
  int parent,
  const std::vector<int> & tokens,
  const BinopPrecedence & prec,
  parse_tree_t & tree,
  int tok)
{
  auto node = tree.addNode(tok, ast_fn_anon, -1);
  parse_expr(node, tokens, prec, tree, tok);
  // TODO Make an anonymous proto.
  //auto Proto = std::make_unique<PrototypeAST>( Identifier{"__anon_expr", FnLoc} );
}



//==============================================================================
// Main parse function
//==============================================================================
parse_tree_t parse(
  const std::vector<int> & tokens,
  const BinopPrecedence & prec)
{
  parse_tree_t res;

  auto ntokens = tokens.size();
  size_t i=0;

  while (i < ntokens) {

    auto tok = tokens[i];
    ++i;

    if (tok == tok_eof)
      return res;

    switch (tok) {
    case tok_sep: // ignore top-level semicolons.
      continue;
    case tok_task:
    case tok_function:
      //parse_function(tokens);
      return res;
    default:
      parse_top_level(i, tokens, prec, res, i-1);
      return res;
    }

  }

  return res;
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

  if (CurTok_ != ')') {
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
  if (CurTok_ == '(') {
    auto ArgsBeginLoc = getCurLoc();
    getNextToken(); // eat (
    std::unique_ptr<NodeAST> Args;
    if (CurTok_ != ')') Args = parseExpression();
    if (CurTok_ != ')')
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
    if (CurTok_ == tok_identifier && isType(Id.getName())) {
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
      if (CurTok_ != ']')
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
    if (CurTok_ == '{') {
      getNextToken(); // eat {
      while (CurTok_ != '}') {
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
    if (CurTok_ == '{') {
      getNextToken(); // eat {
      while (CurTok_ != '}') {
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
    if (CurTok_ == '{') {
      getNextToken(); // eat {
      while (CurTok_ != '}') {
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

  if (CurTok_ != tok_identifier)
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
  if (CurTok_ == '{') {
    getNextToken(); // eat {
    while (CurTok_ != '}') {
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
  case tok_identifier:
    return parseIdentifierExpr();
  case tok_real_literal:
    return parseRealExpr();
  case tok_int_literal:
    return parseIntegerExpr();
  case '(':
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
  case tok_string_literal:
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

  if (CurTok_ == ',') {
    ASTBlock Exprs;
    Exprs.emplace_back( std::move(LHS) );
    while (CurTok_ == ',') {
      getNextToken(); // eat ,
      std::unique_ptr<NodeAST> LHS = parseUnary();
      LHS = parseBinOpRHS(0, std::move(LHS));
      Exprs.emplace_back( std::move(LHS) );
    }
    LHS = std::make_unique<ExprListAST>(
        getLocationRange(BeginLoc),
        std::move(Exprs));
  }
  else if (CurTok_ == ':') {
    ASTBlock Exprs;
    Exprs.emplace_back( std::move(LHS) );
    while (CurTok_ == ':') {
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
  if (!isTokOperator() || CurTok_ == '(' || CurTok_ == ',') {
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

  while (CurTok_ != ':') {
    auto RangeLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'use'.", RangeLoc);
    RangeIds.emplace_back( getIdentifierStr(), RangeLoc );
    getNextToken(); // eat identifier.
    if (CurTok_ == ',') getNextToken(); // eat ,
  }

  auto ColonLoc = getCurLoc();
  if (CurTok_ != ':')
    THROW_SYNTAX_ERROR(
        "Expected ':'.",
        getIdentifierLoc());
  getNextToken(); // eat ":".

  if (CurTok_ != tok_identifier)
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

  while (CurTok_ != ':') {
    auto VarLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'reduce'.", VarLoc);
    VarIds.emplace_back( getIdentifierStr(), VarLoc );
    getNextToken(); // eat identifier.
    if (CurTok_ == ',') getNextToken(); // eat ,
  }

  if (CurTok_ != ':')
    THROW_SYNTAX_ERROR(
        "Expected ':'.",
        getIdentifierLoc());
  getNextToken(); // eat ":".

  if (!isTokOperator() && (CurTok_ != tok_identifier))
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
    
  if (CurTok_ == ';') {
    getNextToken(); // eat ;
    SizeExpr = parseExpression();
  }

  if (CurTok_ != ']')
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
  if (CurTok_ == '{') {
    getNextToken(); // eat {
    while (CurTok_ != '}') {
      if (CurTok_ == tok_return) {
        getNextToken(); // eat return
        Return = parseExpression();
        break;
      }
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }
    if (CurTok_ != '}')
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
  case tok_identifier:
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
    if (CurTok_ == tok_int_literal) {
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
  if (CurTok_ == ',' || CurTok_ == tok_identifier) {
    ReturnTypes.emplace_back(FnName, FnLoc);
    while (CurTok_ == ',') {
      getNextToken(); // eat ,
      if (CurTok_ != tok_identifier)
        THROW_SYNTAX_ERROR(
            "Expected identifier in return type specification.",
            getLocationRange(BeginLoc));
      ReturnTypes.emplace_back(getIdentifierStr(), getIdentifierLoc());
      getNextToken(); // eat identifier
    }
      
    if (CurTok_ != tok_identifier) {
        THROW_SYNTAX_ERROR(
            "Expected function name specification.",
            getLocationRange(BeginLoc));
    }
    FnName = getIdentifierStr();
    FnLoc = getIdentifierLoc();
    getNextToken();
  }

  
  if (CurTok_ != '(')
    THROW_SYNTAX_ERROR(
        "Expected '(' in prototype",
        getIdentifierLoc());

  getNextToken(); // eat "("

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  while (CurTok_ == tok_identifier) {

    bool IsArray = false;

    auto BeginLoc = getCurLoc();

    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR(
          "Identifier expected n prototype for function '" << FnName << "'",
          getLocationRange(BeginLoc));

    auto TypeLoc = getIdentifierLoc();
    auto TypeName = getIdentifierStr();
    ArgTypes.emplace_back( TypeName, TypeLoc );

    getNextToken(); // eat identifier
    
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR(
          "Mising type or variable name in prototype for function '" 
          << FnName << "'",
          getLocationRange(BeginLoc));
    auto VarName = getIdentifierStr();
    auto VarLoc = getIdentifierLoc();
    
    Args.emplace_back( VarName, VarLoc );
    
    getNextToken(); // eat identifier
    
    if (CurTok_ == '[') {
      IsArray = true;
      auto BeginLoc = getCurLoc();
      getNextToken(); // eat the '['.
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR(
            "Expected ']'",
            getLocationRange(BeginLoc));
      getNextToken(); // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (CurTok_ == ',') getNextToken(); // eat ','
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

} // namespace
