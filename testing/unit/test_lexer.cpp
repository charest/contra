#include <contra/lexer.hpp>
#include <contra/token.hpp>
#include <contra/toks.hpp>
#include <contra/stream.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace contra;
using testing::ElementsAre;


//=============================================================================
/// Parameterized test case
//=============================================================================
class LexerTestF : public ::testing::TestWithParam<std::tuple<std::string, int>>
{
public:
  static Tokens toks_;

  static void SetUpTestSuite()
  {
    toks_ = make_contra_tokens();
  }
 
};

Tokens LexerTestF::toks_;

//=============================================================================
// Parameterized Tests
//=============================================================================
TEST_P(LexerTestF, OneTok) {
  auto tok = std::get<1>(GetParam());
  std::cout << "testing tok="<< tok << std::flush;
  auto str = toks_.findInAll(tok);
  std::cout << ", str='" << str << "'" << std::endl;
  ASSERT_GT(str.size(), 0);
  std::istringstream in(str);
  stream_t is(in);
  lexed_t res;
  auto err = lex(toks_, is, res);
  std::cout << "... got " << toks_.findInAll(res.tokens[0]) << std::endl;
  ASSERT_EQ(res.tokens[0], tok);
  ASSERT_FALSE(err);
}

INSTANTIATE_TEST_SUITE_P(
  lexer,
  LexerTestF,
  ::testing::Values(
      std::make_tuple( "comment",  tok_comment ),
      std::make_tuple( "sep",      tok_sep ),
      std::make_tuple( "comma",    tok_comma ),
      std::make_tuple( "colon",    tok_colon ),
      std::make_tuple( "asgmt",    tok_asgmt ),
      std::make_tuple( "lt",       tok_lt ),
      std::make_tuple( "gt",       tok_gt ),
      std::make_tuple( "add",      tok_add ),
      std::make_tuple( "sub",      tok_sub ),
      std::make_tuple( "mul",      tok_mul ),
      std::make_tuple( "div",      tok_div ),
      std::make_tuple( "mod",      tok_mod ),
      std::make_tuple( "lparens",  tok_lparens ),
      std::make_tuple( "rparens",  tok_rparens ),
      std::make_tuple( "lbrack",   tok_lbrack ),
      std::make_tuple( "rbrack",   tok_rbrack ),
      std::make_tuple( "lbrace",   tok_lbrace ),
      std::make_tuple( "rbrace",   tok_rbrace ),
      std::make_tuple( "eq",       tok_eq ),
      std::make_tuple( "ne",       tok_ne ),
      std::make_tuple( "le",       tok_le ),
      std::make_tuple( "ge",       tok_ge ),
      std::make_tuple( "asgn_add", tok_asgmt_add ),
      std::make_tuple( "asgn_sub", tok_asgmt_sub ),
      std::make_tuple( "asgn_mul", tok_asgmt_mul ),
      std::make_tuple( "asgn_div", tok_asgmt_div ),
      std::make_tuple( "if",       tok_if ),
      std::make_tuple( "elif",     tok_elif ),
      std::make_tuple( "else",     tok_else ),
      std::make_tuple( "for",      tok_for ),
      std::make_tuple( "foreach",  tok_foreach ),
      std::make_tuple( "break",    tok_break ),
      std::make_tuple( "reduce",   tok_reduce ),
      std::make_tuple( "use",      tok_use ),
      std::make_tuple( "true",     tok_true ),
      std::make_tuple( "false",    tok_false ),
      std::make_tuple( "function", tok_function ),
      std::make_tuple( "return",   tok_return ),
      std::make_tuple( "task",     tok_task )
  ),
  [](const testing::TestParamInfo<LexerTestF::ParamType>& info)
  { return std::get<0>(info.param); }
);



//=============================================================================
// Individual tests
//=============================================================================

TEST_F(LexerTestF, function_add)
{
  std::stringstream ss;
  ss << "fn sum(i64 a, i64 b) return a+b";

  stream_t is(ss);
  lexed_t res;
  auto err = lex(toks_, is, res);
  print(std::cout, toks_, res);
  EXPECT_THAT( res.tokens, ElementsAre(
    tok_function,
    tok_ident,
    tok_lparens,
    tok_i64,
    tok_ident,
    tok_comma,
    tok_i64,
    tok_ident,
    tok_rparens,
    tok_return,
    tok_ident,
    tok_add,
    tok_ident,
    tok_eof));
  ASSERT_FALSE(err);
  

  EXPECT_EQ(res.numIdentifiers(), 3);
  EXPECT_EQ(res.getIdentifierString(0), "sum");
  EXPECT_EQ(res.getIdentifierString(1), "a");
  EXPECT_EQ(res.getIdentifierString(2), "b");
}

TEST_F(LexerTestF, error)
{
  std::stringstream ss;
  ss << "0..1";
 
  stream_t is(ss);
  lexed_t res;
  auto err = lex(toks_, is, res);
  ASSERT_TRUE(err);
}
