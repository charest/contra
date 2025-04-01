#include <contra/lexer.hpp>
#include <contra/token.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace contra;

TEST(lexer, simple)
{
  Tokens::setup();
  std::stringstream ss;
  ss << "fn sum(i64 a, i64 b) return a+b";
 
  auto res = lex(ss);
}
