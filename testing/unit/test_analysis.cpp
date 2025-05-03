#include <contra/graph.hpp>
#include <contra/lexer.hpp>
#include <contra/parser.hpp>
#include <contra/precedence.hpp>
#include <contra/stream.hpp>
#include <contra/token.hpp>
#include <contra/toks.hpp>

#include <gtest/gtest.h>

#include <iomanip>
#include <stack>

using namespace contra;

void visit(const lexed_t & lx, parse_tree_t & tr, graph_t & gr)
{
  
  std::stack<std::tuple<int,int,int>> q;
  
  for (auto r : gr.roots) {
    auto dty = tr.findType(r);
    q.push({r,dty, 1});
  }
    
  while (q.size()) {
    auto cur = q.top();
    auto i       = std::get<0>(cur);
    auto par_dty = std::get<1>(cur);
    auto dir     = std::get<2>(cur);
    q.pop();
    
    auto ty = tr.node_ast_type[i];
    auto tid = tr.node_to_token[i];
    auto tok = lx.tokens[tid];
    
    if (dir > 0)
      std::cout << " -> ";
    else
      std::cout << " <- ";
    std::cout << "n: " << i << " " << tok_to_string(ty) << " " << tok_to_string(tok);
    auto dty = tr.findType(i);
    std::cout << "    l: " << par_dty << " r: " << dty << std::endl;
    
    switch (ty) {
    case AST_FN_DEF:{
      auto id = lx.findIdentifier(tid);
      auto str = lx.getIdentifierString(id);
      std::cout << "FUNC: " << str << std::endl;
      tr.addFunc(tid);
      break;
    }
    case AST_FN_ARG:{
      auto id = lx.findIdentifier(tid);
      auto str = lx.getIdentifierString(id);
      std::cout << "PARAM: " << str << std::endl;
      break;
    }
    }
       

    if (dir > 0) 
    { 
      auto nc = gr.size(i);
      if (nc) q.push({i, dty, -1});

      for (int c=nc; c-->0; ) {
        auto neigh = gr(i,c);
        q.push({neigh, dty, dir});
      }

    }

  }
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

  static void test(const std::string & str)
  {
    std::stringstream ss;
    ss << str;
  
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "| Testing: " << ss.str() << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // build the contra ast and graph
    stream_t is(ss);
    lexed_t lx;
    parse_tree_t tree;
    ASSERT_FALSE( lex(is, toks_, lx) );
    ASSERT_FALSE( parse(is, lx, prec_, tree) );
    auto gr = graph(tree.node_parent);
    
    print(std::cout, lx);
    print(std::cout, tree, gr);

    visit(lx, tree, gr);
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
}
