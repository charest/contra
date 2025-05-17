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
    auto is = make_stream(ss);
    lexed_t res;
    auto err = lex(is, res);
    
    EXPECT_EQ(res.size(), 2);
    auto tok = res.tokens[0];
    EXPECT_EQ(tok, ans_tok);
    std::cout << "... Expected: " << tok_to_string(tok);
    std::cout << " Got: " << tok_to_string(tok) << std::endl;
    
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
    auto is = make_stream(ss);
    lexed_t res;
    auto err = lex(is, res);
    
    auto nans = ans.size();
    EXPECT_EQ(res.size(), nans+1);
    
    for (int i=0; i<nans; ++i) {
      auto exp_tok = ans[i].first;
      auto & exp_id = ans[i].second;
      auto tok = res.tokens[i];
      EXPECT_EQ(tok, exp_tok);
      std::cout << "... [" << i << "] Expected: " << tok_to_string(tok);
      std::cout << " Got: " << tok_to_string(tok) << std::endl;
      auto pos = res.token_pos[i];
      auto str = is.buffer.substr(pos.begin, pos.length());
      EXPECT_EQ(str, exp_id);
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
  test_w_ident("ident",  {{TOK_IDENT, "ident"}});
  test_w_ident("id1ent", {{TOK_IDENT, "id1ent"}});
  test_w_ident("1ident", {{TOK_INT_LIT, "1"}, {TOK_IDENT, "ident"}});
}


TEST_F(LexerTestF, quote) {
  test_w_ident("\"Quoted\"", {{TOK_STRING_LIT, "\"Quoted\""}});
}

TEST_F(LexerTestF, comment) {
  test_w_ident("# test", {{TOK_COMMENT, "# test"}});
  test_w_ident("# test\nident", {{TOK_COMMENT, "# test"}, {TOK_IDENT, "ident"}});
}

TEST_F(LexerTestF, number) {
  test_w_ident("1"      , {{TOK_INT_LIT , "1"      }});
  test_w_ident("12"     , {{TOK_INT_LIT , "12"     }});
  test_w_ident("1.2"    , {{TOK_REAL_LIT, "1.2"    }});
  test_w_ident(".2"     , {{TOK_REAL_LIT, ".2"     }});
  test_w_ident("0.2"    , {{TOK_REAL_LIT, "0.2"    }});
  test_w_ident("1.2e5"  , {{TOK_REAL_LIT, "1.2e5"  }});
  test_w_ident("1.2E5"  , {{TOK_REAL_LIT, "1.2E5"  }});
  test_w_ident("1.2e-5" , {{TOK_REAL_LIT, "1.2e-5" }});
  test_w_ident("1.2e+5" , {{TOK_REAL_LIT, "1.2e+5" }});
  test_w_ident("1.2e+55", {{TOK_REAL_LIT, "1.2e+55"}});
  
  test_w_ident("1..2" , {{TOK_REAL_LIT,"1..2" }}, true);
  test_w_ident("1...2", {{TOK_REAL_LIT,"1...2"}}, true);
  test_w_ident("1.2e+", {{TOK_REAL_LIT,"1.2e+"}}, true);
  test_w_ident("1.2ee", {{TOK_REAL_LIT,"1.2ee"}}, true);
  test_w_ident("1.2e ", {{TOK_REAL_LIT,"1.2e "}}, true);
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

  auto is = make_stream(ss);
  lexed_t res;
  auto err = lex(is, res);
  recognize(is, toks_, res);
  print(std::cout, is, res);
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
}

TEST_F(LexerTestF, error)
{
  std::stringstream ss;
  ss << "0..1";
 
  auto is = make_stream(ss);
  lexed_t res;
  auto err = lex(is, res);
  ASSERT_TRUE(err);
}
