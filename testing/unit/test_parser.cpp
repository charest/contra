#include <contra/graph.hpp>
#include <contra/lexer.hpp>
#include <contra/parser.hpp>
#include <contra/precedence.hpp>
#include <contra/token.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace contra;
using testing::ElementsAre;

// Parameterized test case
class ParseTestF : public ::testing::TestWithParam<std::tuple<std::string, int>>
{
public:
  static Tokens toks_;
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
  
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Testing: " << ss.str() << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    auto lex_res = lex(toks_, ss);
    auto parse_tree = parse(lex_res.tokens, prec_);
    auto gr = graph(parse_tree.node_parent);
    print(std::cout, toks_, lex_res);
    print(std::cout, parse_tree, gr);
    print(std::cout, toks_, lex_res, parse_tree, gr);
  }

 
};

Tokens ParseTestF::toks_;
BinopPrecedence ParseTestF::prec_;

TEST_F(ParseTestF, unary)
{
  test("+a");
}

TEST_F(ParseTestF, binop)
{
  test("a+b");
  test("a+b+c+d");
  test("a+b*c");
  test("a*b+c");
  test("a+b,c-d,e*f");
  test("a+b:c-d:e*f");
  test("a=b+c*d");
}

TEST_F(ParseTestF, function)
{
  test("fn sum(i64 a, i64 b) a+b");
}
