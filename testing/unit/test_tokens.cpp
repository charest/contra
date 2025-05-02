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
  static token_map_t toks_;

  static void SetUpTestSuite()
  {
    toks_ = make_contra_tokens();
  }
 
};

token_map_t TokenTestF::toks_;

void test_map(const token_map_t & tmap) {
  for (auto t : tmap.str_to_enum) {
    auto str = t.first;
    auto tok = t.second;
    std::cout << "Processing " << str << " <--> " << tok << std::endl;
    EXPECT_EQ( tok_to_string(tok), str );
    EXPECT_EQ( tmap.find(str), tok );
  }
}

void test_exact_symbols(char c) {
  token_map_t test;
  test.add(c);
  auto str = std::string(1,c);
  EXPECT_EQ(tok_to_string(c), str);
  EXPECT_EQ(test.find(str), c);
}

void test_str(int tok, const std::string & str) {
  token_map_t test;
  test.add(tok, str);
  EXPECT_EQ(tok_to_string(tok), str);
  EXPECT_EQ(test.find(str), tok);
}

TEST(tokens, map)
{
  test_exact_symbols('a');
  test_str('a', "a");
}

TEST_F(TokenTestF, contra)
{
  test_map(toks_);
}
