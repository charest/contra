#ifndef PARSER_HPP
#define PARSER_HPP

#include "ast.hpp"
#include "context.hpp"
#include "lexer.hpp"

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace contra {

struct BinopPrecedence;
struct stream_t;
struct graph_t;
struct lexed_t;
struct token_map_t;

struct parse_tree_t {
  std::vector<int> node_to_token;
  std::vector<int> node_ast_type;
  
  std::vector<int> node_parent;
 
  std::map<std::vector<int>, int> types;
  std::unordered_map<int,int> node_to_type;
  std::unordered_map<int,int> node_to_type_token;

  size_t size() const { return node_ast_type.size(); }

  int addNode(int tok, int ty, int parent = -1)
  {
    auto id = node_ast_type.size();
    node_to_token.emplace_back(tok);
    node_ast_type.emplace_back(ty);
    node_parent.emplace_back(parent);
    return id;
  }

  void setParent(int node, int parent)
  { node_parent[node] = parent; }
  
  void setNodeType(int node, int ty)
  { node_ast_type[node] = ty; }

  int installType(int ty)
  {
    auto n = types.size();
    auto it = types.emplace(std::vector<int>{ty}, n);
    return it.first->second;
  }
  int installType(const std::vector<int> & tys)
  {
    auto n = types.size();
    auto it = types.emplace(tys, n);
    return it.first->second;
  }

  void setType(int node, int ty)
  { node_to_type[node] = ty; }
  
  void setType(int node, int ty, int tok)
  {
    node_to_type[node] = ty;
    node_to_type_token[node] = tok;
  }

  int findType(int n) {
    auto it = node_to_type.find(n);
    if (it != node_to_type.end()) return it->second;
    return -1;
  }

  //int addFunc(int tid)
  //{
  //  auto id = lx.findIdentifier(tid);
  //  ident_to_func[id] = ident_to_func.size();
  //}
};

/// Parse tokens
int parse(
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & res);
int parse_sext(
  stream_t & is,
  const lexed_t & lx,
  const BinopPrecedence & prec,
  parse_tree_t & res);

/// Dump results in tabular form
void print(std::ostream& os, const parse_tree_t & tree, const graph_t & graph);

/// Dump results in sext form
void print(
  std::ostream& os,
  const stream_t & stream,
  const lexed_t & lex,
  const parse_tree_t & tree,
  const graph_t & graph);

/// Compare two trees
bool compare(
  const stream_t & isa,
  const lexed_t & lxa,
  const parse_tree_t &tra,
  const graph_t & gra,
  const stream_t & isb,
  const lexed_t & lxb,
  const parse_tree_t &trb,
  const graph_t & grb);

} // namespace

#endif // PARSER_HPP
