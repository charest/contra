#include <contra/token.hpp>
#include <contra/toks.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace contra;
using testing::ElementsAre;

// Parameterized test case
class TokenTestF : public testing::Test
{
public:
  static Tokens toks_;

  static void SetUpTestSuite()
  {
    toks_ = make_contra_tokens();
  }
 
};

Tokens TokenTestF::toks_;

void test_set(const Tokens& toks, const token_set_t & tset) {
  for (auto tok : tset.tokens) {
    auto as_str = std::string(1, tok);
    std::cout << "Processing " << tok << std::endl;
    EXPECT_EQ( tset.find(tok), tok );
    EXPECT_EQ( toks.findInAll(tok), as_str );
    EXPECT_EQ( toks.findInAll(as_str), tok );
  }
}


void test_map(const Tokens& toks, const token_map_t & tmap) {
  for (auto t : tmap.enum_to_str) {
    auto tok = t.first;
    auto str = t.second;
    std::cout << "Processing " << tok << " <--> " << str << std::endl;
    EXPECT_EQ( tmap.find(tok), str );
    EXPECT_EQ( tmap.find(str), tok );
    EXPECT_EQ( toks.findInAll(tok), str );
    EXPECT_EQ( toks.findInAll(str), tok );
  }
  for (auto t : tmap.str_to_enum) {
    auto str = t.first;
    auto tok = t.second;
    std::cout << "Processing " << str << " <--> " << tok << std::endl;
    EXPECT_EQ( tmap.find(tok), str );
    EXPECT_EQ( tmap.find(str), tok );
    EXPECT_EQ( toks.findInAll(tok), str );
    EXPECT_EQ( toks.findInAll(str), tok );
  }
}

void test_exact_symbols(char c) {
  token_map_t test;
  test.add(c);
  auto str = std::string(1,c);
  EXPECT_EQ(test.find(c), str);
  EXPECT_EQ(test.find(str), c);
}

void test_str(int tok, const std::string & str) {
  token_map_t test;
  test.add(tok, str);
  EXPECT_EQ(test.find(tok), str);
  EXPECT_EQ(test.find(str), tok);
}

TEST(tokens, map)
{
  test_exact_symbols('a');
  test_str(1, "a");
}

TEST_F(TokenTestF, contra)
{
  test_set(toks_, toks_.exact_symbols);
  test_map(toks_, toks_.inexact_symbols);
  test_map(toks_, toks_.keywords);
  test_map(toks_, toks_.types);
  test_map(toks_, toks_.tags);
}
