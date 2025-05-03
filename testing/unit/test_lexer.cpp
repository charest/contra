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
class LexerTestF : public ::testing::Test
{
public:
  static token_map_t toks_;

  static void SetUpTestSuite()
  {
    toks_ = make_contra_tokens();
  }

  //---------------------------------------------------------------------------
  void test(const std::string & ans, int ans_tok, bool isBad=false)
  {
    std::cout << "Testing: " << ans << std::endl;
  
    std::stringstream ss(ans);
    stream_t is(ss);
    lexed_t res;
    auto err = lex(is, toks_, res);
    
    if (!isBad) EXPECT_EQ(res.numTokens(), 2);
    auto tok = res.tokens[0];
    if (!isBad) EXPECT_EQ(tok, ans_tok);
    if (!isBad) EXPECT_EQ(res.tokens[1], TOK_EOF);
    std::cout << "... Expected: " << tok_to_string(tok);
    std::cout << " Got: " << tok_to_string(tok) << std::endl;
    
    if (isBad) ASSERT_TRUE(err);
    else       ASSERT_FALSE(err);
  }
  
  
  //---------------------------------------------------------------------------
  void test_w_ident(const std::string & ans, int ans_tok, bool isBad=false)
  {
    std::cout << "Testing: " << ans << std::endl;
  
    std::stringstream ss(ans);
    stream_t is(ss);
    lexed_t res;
    auto err = lex(is, toks_, res);
    
    if (!isBad) EXPECT_EQ(res.numTokens(), 2);
    auto tok = res.tokens[0];
    if (!isBad) EXPECT_EQ(tok, ans_tok);
    if (!isBad) EXPECT_EQ(res.tokens[1], TOK_EOF);
    std::cout << "... Expected: " << tok_to_string(tok);
    std::cout << " Got: " << tok_to_string(tok) << std::endl;
    
    if (!isBad) EXPECT_EQ(res.numIdentifiers(), 1);
    auto id = res.findIdentifier(0);
    auto str = res.getIdentifierString(id);
    std::cout << "... Ident: " << str << std::endl;
  
    EXPECT_EQ(str, ans);
    if (isBad) ASSERT_TRUE(err);
    else       ASSERT_FALSE(err);
  }
  
  //---------------------------------------------------------------------------
  void test_w_ident(
    const std::string & inp,
    const std::vector<std::pair<int, std::string>> & ans,
    bool isBad=false)
  {
    std::cout << "Testing: " << inp << std::endl;
  
    std::stringstream ss(inp);
    stream_t is(ss);
    lexed_t res;
    auto err = lex(is, toks_, res);
    
    auto nans = ans.size();
    if (!isBad) EXPECT_EQ(res.numTokens(), nans+1);
    if (!isBad) EXPECT_EQ(res.tokens.back(), TOK_EOF);
    
    for (int i=0; i<nans; ++i) {
      auto exp_tok = ans[i].first;
      auto & exp_id = ans[i].second;
      auto tok = res.tokens[i];
      if (!isBad) EXPECT_EQ(tok, exp_tok);
      std::cout << "... [" << i << "] Expected: " << tok_to_string(tok);
      std::cout << " Got: " << tok_to_string(tok) << std::endl;
      auto id = res.findIdentifier(i);
      auto str = res.getIdentifierString(id);
      if (!isBad) EXPECT_EQ(str, exp_id);
      std::cout << "... [" << i << "] Expected: " << exp_id;
      std::cout << " Got: " << str << std::endl;
    }
  
    if (isBad) ASSERT_TRUE(err);
    else       ASSERT_FALSE(err);
  }
 
 
};

token_map_t LexerTestF::toks_;

//=============================================================================
// Individual tests
//=============================================================================

TEST_F(LexerTestF, ident) {
  test_w_ident("ident", TOK_IDENT);
  test_w_ident("id1ent", TOK_IDENT);
  test_w_ident("1ident", {{TOK_INT_LIT, "1"}, {TOK_IDENT, "ident"}});
}


TEST_F(LexerTestF, quote) {
  test_w_ident("\"Quoted\"", {{TOK_STRING_LIT, "Quoted"}});
}

TEST_F(LexerTestF, comment) {
  test_w_ident("# test", {{TOK_COMMENT, ""}});
  test_w_ident("# test\nident", {{TOK_COMMENT, ""}, {TOK_IDENT, "ident"}});
}

TEST_F(LexerTestF, number) {
  test_w_ident("1"      , TOK_INT_LIT);
  test_w_ident("12"     , TOK_INT_LIT);
  test_w_ident("1.2"    , TOK_REAL_LIT);
  test_w_ident(".2"     , TOK_REAL_LIT);
  test_w_ident("0.2"    , TOK_REAL_LIT);
  test_w_ident("1.2e5"  , TOK_REAL_LIT);
  test_w_ident("1.2E5"  , TOK_REAL_LIT);
  test_w_ident("1.2e-5" , TOK_REAL_LIT);
  test_w_ident("1.2e+5" , TOK_REAL_LIT);
  test_w_ident("1.2e+55", TOK_REAL_LIT);
  
  test_w_ident("1..2",  TOK_REAL_LIT, true);
  test_w_ident("1...2", TOK_REAL_LIT, true);
  test_w_ident("1.2e+", TOK_REAL_LIT, true);
  test_w_ident("1.2ee", TOK_REAL_LIT, true);
  test_w_ident("1.2e ", TOK_REAL_LIT, true);
}

TEST_F(LexerTestF, ops) {
  test("+" , '+');
  test("+=", TOK_ADD_EQ);
  test("-" , '-');
  test("-=", TOK_SUB_EQ);
  test("*" , '*');
  test("*=", TOK_MUL_EQ);
  test("=" , '=');
  test("==", TOK_EQUIV);
  test("!" , '!');
  test("!=", TOK_NE);
  test("<" , '<');
  test("<=", TOK_LE);
  test(">" , '>');
  test(">=", TOK_GE);
}

TEST_F(LexerTestF, punc) {
  test("," , ',');
  test(";" , ';');
  test("." , '.');
  test("%" , '%');
}

TEST_F(LexerTestF, function_add)
{
  std::stringstream ss;
  ss << "fn sum(i64 a, i64 b) return a+b";

  stream_t is(ss);
  lexed_t res;
  auto err = lex(is, toks_, res);
  print(std::cout, res);
  EXPECT_THAT( res.tokens, ElementsAre(
    TOK_FUNC,
    TOK_IDENT,
    '(',
    TOK_I64,
    TOK_IDENT,
    ',',
    TOK_I64,
    TOK_IDENT,
    ')',
    TOK_RETURN,
    TOK_IDENT,
    '+',
    TOK_IDENT,
    TOK_EOF));
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
  auto err = lex(is, toks_, res);
  ASSERT_TRUE(err);
}
