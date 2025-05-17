#include <contra/graph.hpp>
#include <contra/lexer.hpp>
#include <contra/parser.hpp>
#include <contra/precedence.hpp>
#include <contra/stream.hpp>
#include <contra/token.hpp>
#include <contra/toks.hpp>

#include <gtest/gtest.h>

#include <forward_list>
#include <iomanip>
#include <stack>

using namespace contra;

struct ident_map_t {
  std::unordered_map<std::string, int> identifier_to_id;
  std::vector<std::string> identifiers;
  
  int insert(const std::string & str) {
    auto n = identifiers.size();
    // try to insert the identifier
    auto res = identifier_to_id.try_emplace( str, n );
    // if new, add it to the vector as well
    if (res.second) identifiers.emplace_back( str );
    return res.first->second;
  }

  std::string get(int i) const
  {
    if (i<0 || i > identifiers.size()) return "";
    return identifiers[i];
  }

};

using symbol_t = std::unordered_map<int, int>;
using scoped_symbol_t = std::forward_list<symbol_t>;

int find(const symbol_t & tab, int key) {
  auto it = tab.find(key);
  if (it != tab.end()) return it->first;
  return -1;
}

void insert(symbol_t & tab, int key, int val)
{ tab.emplace(key, val); }

int find(const scoped_symbol_t & scoped_tab, int key) {
  if (scoped_tab.empty())
    throw std::runtime_error("Empty symbol table!");
  for (auto & tab : scoped_tab) {
    auto it = tab.find(key);
    if (it != tab.end()) return it->first;
  }
  return -1;
}

void insert(scoped_symbol_t & tab, int key, int val)
{ tab.front().emplace(key, val); }


int visit(const stream_t & stream, const lexed_t & lx, parse_tree_t & tr, graph_t & gr)
{
  ident_map_t ident_map;
  std::unordered_map<int,int> token_to_ident; 

  for (size_t i=0; i<lx.tokens.size(); ++i) {
    auto tok = lx.tokens[i];
    
    switch (tok) {
    case TOK_IDENT: {
      auto pos = lx.token_pos[i];
      auto str = stream.at(pos);
      auto id = ident_map.insert(str);
      token_to_ident.emplace(i, id);
    }};

  }
 
  int err = 0;
  scoped_symbol_t symtab;
  symbol_t funtab;
  std::unordered_map<int,int> node_to_def;

  std::stack<std::tuple<int,int,int>> q;
  
  for (auto r : gr.roots) q.push({r, 1, 0});
  symtab.push_front({});
    
  while (q.size()) {
    auto cur = q.top();
    auto i     = std::get<0>(cur);
    auto dir   = std::get<1>(cur);
    auto depth = std::get<2>(cur);
    q.pop();
    

    // Visit
    auto ty = tr.node_ast_type[i];
    auto tid = tr.node_to_token[i];
    auto pos = lx.token_pos[tid];
    auto str = stream.at(pos);
    
    
    switch (ty) {
    case AST_FOR:
    case AST_IF:
    case AST_BLOCK:
      if (dir>0) symtab.push_front({});
      else       symtab.pop_front();
      break;
    
    case AST_FN_DEF:
    if (dir > 0) {
      // find identifier id, if there check it
      auto it = token_to_ident.find(tid);
      if (it != token_to_ident.end()) {
        // check symbol table, if already there, error
        auto ident = it->second;
        auto i_def = find(funtab, ident);
        if (i_def != -1) {
          auto tid_def = tr.node_to_token[i_def];
          auto pos_def = lx.token_pos[tid_def];
          err += error(stream, "Redefinition of previously defined variable.", pos);
          error(stream, "First defined here.", pos_def);
        }
        // if not there, add it
        else {
          insert(funtab, ident, i);
        }
      }
      // else if no identifier id, error
      else {
        err += error(stream, "Internal error. Couldn't find identifier hash.", pos);
      }

      symtab.push_front({});

    }
    else {
      symtab.pop_front();
    }
      
    break;
    

    case AST_FN_ARG:
    {
      // find identifier id, if there check it
      auto it = token_to_ident.find(tid);
      if (it != token_to_ident.end()) {
        // check symbol table, if already there, error
        auto ident = it->second;
        auto i_def = find(symtab, ident);
        if (i_def != -1) {
          auto tid_def = tr.node_to_token[i_def];
          auto pos_def = lx.token_pos[tid_def];
          err += error(stream, "Redefinition of previously defined variable.", pos);
          error(stream, "First defined here.", pos_def);
        }
        // if not there, add it
        else {
          insert(symtab, ident, i);
        }
      }
      // else if no identifier id, error
      else {
        err += error(stream, "Internal error. Couldn't find identifier hash.", pos);
      }

      break;
    } // case
    
    case AST_VAR: if (dir>0) {
      auto it = token_to_ident.find(tid);
      if (it != token_to_ident.end()) {
        // check symbol table, if not there, error.  otherwise, use it
        auto i_def = find(symtab, it->second);
        if (i_def == -1) err += error(stream, "Use of undefined variable.", pos);
        else node_to_def.emplace(i, i_def);
      }
      // else if no identifier id, error
      else {
        err += error(stream, "Internal error. Couldn't find identifier hash.", pos);
      }
      break;
    } // case
    
    case AST_FN_CALL: if (dir>0) {
      auto it = token_to_ident.find(tid);
      if (it != token_to_ident.end()) {
        // check symbol table, if not there, error.  otherwise, use it
        auto i_def = find(funtab, it->second);
        if (i_def == -1) err += error(stream, "Function not defined.", pos);
        else node_to_def.emplace(i, i_def);
      }
      // else if no identifier id, error
      else {
        err += error(stream, "Internal error. Couldn't find identifier hash.", pos);
      }
      break;
    } // case

    } // switch
      
    // add children
    if (dir > 0) 
    { 
      auto nc = gr.size(i);
      if (nc) q.push({i, -1, depth});
      for (int c=nc; c-->0; ) {
        auto neigh = gr(i,c);
        q.push({neigh, dir, depth+1});
      }
    }

  } // while

  symtab.pop_front();

  return err;
}

//=============================================================================
/// Parameterized test case
//=============================================================================
class AnalysisTestF : public ::testing::Test
{
public:
  static token_map_t toks_;
  static BinopPrecedence prec_;

  static void SetUpTestSuite()
  {
    toks_ = make_contra_tokens();
    prec_ = make_contra_precedence();
  }

  static void test(const std::string & str, bool isbad = false)
  {
    std::stringstream ss;
    ss << str;
  
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "| Testing: " << ss.str() << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // build the contra ast and graph
    auto is = make_stream(ss);
    lexed_t lx;
    parse_tree_t tree;
    ASSERT_FALSE( lex(is, lx) );
    recognize(is, toks_, lx);
    ASSERT_FALSE( parse(is, lx, prec_, tree) );
    auto gr = graph(tree.node_parent);
    
    print(std::cout, is, lx);
    print(std::cout, tree, gr);
    print(std::cout, is, lx, tree, gr);

    auto err = visit(is, lx, tree, gr);
    if (isbad) ASSERT_TRUE (err);
    else       ASSERT_FALSE(err);
  }
 
};

token_map_t AnalysisTestF::toks_;
BinopPrecedence AnalysisTestF::prec_;

//=============================================================================
// TESTS
//=============================================================================

TEST_F(AnalysisTestF, simple)
{
  test("fn sum(i64 a, i64 b) return a+b");
  test("fn sum(i64 a, i64 b) return a+c", true);
  test("fn sum(i64 a, i64 a) return a+b", true);
}

TEST_F(AnalysisTestF, assign)
{
  test("a = 0");
  test("a = b", true);
}

TEST_F(AnalysisTestF, ifstmt)
{
  test(
"fn sum(i64 a, i64 b) {\n"
"  if a==b\n"
"    a = 3\n" 
"return a+b}");
  test(
"fn sum(i64 a, i64 b) {\n"
"  if a==b\n"
"    c = 3\n"
"  else\n"
"    d = c\n"
"return a+b}");
}

TEST_F(AnalysisTestF, fun)
{
  test("fn sum(i64 a, i64 b) sum(a, b)");
  test("fn sum(i64 a, i64 b) add(a, b)", true);
}
