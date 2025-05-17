#include <contra/graph.hpp>
#include <contra/lexer.hpp>
#include <contra/parser.hpp>
#include <contra/precedence.hpp>
#include <contra/stream.hpp>
#include <contra/token.hpp>
#include <contra/toks.hpp>

#include <gtest/gtest.h>

#include <iomanip>

using namespace contra;

//=============================================================================
/// Parameterized test case
//=============================================================================
class ParseTestF : public ::testing::Test
{
public:
  static token_map_t toks_, sext_toks_;
  static BinopPrecedence prec_;

  static void SetUpTestSuite()
  {
    toks_ = make_contra_tokens();
    prec_ = make_contra_precedence();
    sext_toks_ = make_sext_tokens();
  }

  static void test(const std::string & str, const std::string & ans)
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

    std::cout << "| AST" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    print(std::cout, is, lx);
    print(std::cout, tree, gr);
    print(std::cout, is, lx, tree, gr);
    
    
    //std::cout << std::string(80, '-') << std::endl;
    //std::cout << "| SEXT (AST)" << std::endl;
    //std::cout << std::string(80, '-') << std::endl;
    
    // build ast/graph from provided sext
    std::istringstream in(ans);
    auto sext_is = make_stream(in);
    lexed_t sext_lx;
    parse_tree_t sext_tree;
    ASSERT_FALSE( lex(sext_is, sext_lx) );
    recognize(sext_is, sext_toks_, sext_lx);
    ASSERT_FALSE( parse_sext(sext_is, sext_lx, prec_, sext_tree) );
    auto sext_gr = graph(sext_tree.node_parent);
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "| SEXT" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    print(std::cout, sext_is, sext_lx);
    print(std::cout, sext_tree, sext_gr);
    print(std::cout, sext_is, sext_lx, sext_tree, sext_gr);

    ASSERT_TRUE( compare(is, lx, tree, gr, sext_is, sext_lx, sext_tree, sext_gr) );
  }

 
};

token_map_t ParseTestF::toks_;
token_map_t ParseTestF::sext_toks_;
BinopPrecedence ParseTestF::prec_;

//=============================================================================
// TESTS
//=============================================================================

TEST_F(ParseTestF, unary)
{
  test("+a", "(Unary + (Var a))");
  //test("*a", "(Unary * (Var a))"); // should die
}

TEST_F(ParseTestF, parens)
{
  test("(a)", "(Var a)");
}

TEST_F(ParseTestF, binop)
{
  test("a+b", "(Binary + (Var a) (Var b))");
  test("a+b+c+d", 
  "(Binary + "
  "  (Binary + "
  "    (Binary + "
  "      (Var a)"
  "      (Var b))"
  "    (Var c))"
  "  (Var d))");
  test("a+b*c", "(Binary + "
  "  (Var a) "
  "  (Binary * "
  "    (Var b) "
  "    (Var c)))");
  test("a*b+c",
  " (Binary + "
  "   (Binary * "
  "     (Var a) "
  "     (Var b)) "
  "   (Var c))");
  test("(a+b) + (c+d)",
  " (Binary + "
  "   (Binary + "
  "     (Var a) "
  "     (Var b)) "
  "   (Binary + "
  "     (Var c) "
  "     (Var d)))");
}

TEST_F(ParseTestF, exprlist)
{
  test("a,b,c",
  "  (ExprList "
  "    (Var a) "
  "    (Var b) "
  "    (Var c))");
  test("a+b,c-d,e*f",
  "  (ExprList "
  "    (Binary + "
  "      (Var a) "
  "      (Var b)) "
  "    (Binary - "
  "      (Var c) "
  "      (Var d)) "
  "    (Binary * "
  "      (Var e) "
  "      (Var f)))");
  test("a+b:c-d:e*f",
  " (Range "
  "   (Binary + "
  "     (Var a) "
  "     (Var b)) "
  "   (Binary - "
  "     (Var c) "
  "     (Var d)) "
  "   (Binary * "
  "     (Var e) "
  "     (Var f)))");
  test("a,b=b+c*d", 
  " (Assign = "
  "   (ExprList "
  "     (Var a) "
  "     (Var b)) "
  "   (Binary + "
  "     (Var b) "
  "     (Binary * "
  "       (Var c) "
  "       (Var d))))");
}

TEST_F(ParseTestF, literal)
{
  test("1.2", "(RealLit 1.2)");
  test("1", "(IntLit 1)");
  test("-1", "(Unary - (IntLit 1))");
  test("\"str\"", "(StringLit \"str\")");
}

TEST_F(ParseTestF, call)
{
  test("test(a)",
  " (FunCall test "
  "   (Var a)) ");
  test("test(a, b)",
  " (FunCall test "
  "   (Var a) "
  "   (Var b))");
}

TEST_F(ParseTestF, var)
{
  test("i64 a = 1",
  " (Assign = "
  "   (Var a) "
  "   (IntLit 1))");
  test("a = 1",
  " (Assign = "
  "   (Var a) "
  "   (IntLit 1))");
}

TEST_F(ParseTestF, array)
{
  test("a[i]",
  " (ArrIndex a "
  "   (Var i)) ");
  test("a = [1; 2]",
  " (Assign = "
  "   (Var a) "
  "   (ArrInit "
  "     (IntLit 1) "
  "     (IntLit 2))) ");
  test("a = [1, 2, 3]",
  " (Assign = "
  "   (Var a) "
  "   (ArrInit "
  "     (ExprList "
  "       (IntLit 1) "
  "       (IntLit 2) "
  "       (IntLit 3))))");
  test("a,b = [1, 2, 3]",
  " (Assign = "
  "   (ExprList "
  "     (Var a) "
  "     (Var b)) "
  "   (ArrInit "
  "     (ExprList "
  "       (IntLit 1) "
  "       (IntLit 2) "
  "       (IntLit 3))))");
}

TEST_F(ParseTestF, ifstmt)
{
  test("if (a==b) x=2", 
  "  (If \n"
  "    (Binary == \n"
  "      (Var a) \n"
  "      (Var b)) \n"
  "    (Assign = \n"
  "      (Var x) \n"
  "      (IntLit 2)))");
  test("if (a==b) x=2 elif (a==c) x=1", 
  " (If \n"
  "   (Binary == \n"
  "     (Var a) \n"
  "     (Var b)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 2)) \n"
  "   (Binary == \n"
  "     (Var a) \n"
  "     (Var c)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 1)))");
  test("if (a==b) x=2 else x=1", 
  " (If \n"
  "   (Binary == \n"
  "     (Var a) \n"
  "     (Var b)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 2)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 1)))");
  test("if (a==b) x=1 elif (a==c) x=2 elif (a==d) x=3 else x=4", 
  " (If \n"
  "   (Binary == \n"
  "     (Var a) \n"
  "     (Var b)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 1)) \n"
  "   (Binary == \n"
  "     (Var a) \n"
  "     (Var c)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 2)) \n"
  "   (Binary == \n"
  "     (Var a) \n"
  "     (Var d)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 3)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (IntLit 4)))");
}

TEST_F(ParseTestF, forstmt)
{
  test("for i = 1:2 x=i",
  " (For \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (Var i)))");
  test("for i = 1:2 { x=a x=b }",
  " (For \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Block \n"
  "     (Assign = \n"
  "       (Var x) \n"
  "       (Var a)) \n"
  "     (Assign = \n"
  "       (Var x) \n"
  "       (Var b))))");
}

TEST_F(ParseTestF, foreach)
{
  test("foreach i = 1:2 x=i",
  " (Foreach \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Assign = \n"
  "     (Var x) \n"
  "     (Var i)))");
}

TEST_F(ParseTestF, use)
{
  test("foreach i = 1:2 { use part : b x=i }",
  " (Foreach \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Block \n"
  "     (Use \n"
  "       (Var part) \n"
  "       (Var b)) \n"
  "     (Assign = \n"
  "       (Var x) \n"
  "       (Var i))))");
  test("foreach i = 1:2 { use p1, p2 : b x=i }",
  " (Foreach \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Block \n"
  "     (Use \n"
  "       (Var p1) \n"
  "       (Var p2) \n"
  "       (Var b)) \n"
  "     (Assign = \n"
  "       (Var x) \n"
  "       (Var i))))");
}

TEST_F(ParseTestF, reduce)
{
  test("foreach i = 1:2 { reduce x : + x=i }",
  " (Foreach \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Block \n"
  "     (Reduce \n"
  "       (Var x) \n"
  "       (ReduceOp +)) \n"
  "     (Assign = \n"
  "       (Var x) \n"
  "       (Var i))))");
  test("foreach i = 1:2 { reduce x,y : max x=i }",
  " (Foreach \n"
  "   (Var i) \n"
  "   (Range \n"
  "     (IntLit 1) \n"
  "     (IntLit 2)) \n"
  "   (Block \n"
  "     (Reduce \n"
  "       (Var x) \n"
  "       (Var y) \n"
  "       (ReduceOp max)) \n"
  "     (Assign = \n"
  "       (Var x) \n"
  "       (Var i))))");
}

TEST_F(ParseTestF, func)
{
  test("fn sum(i64 a, i64 b) a+b",
  "  (FunDef sum \n"
  "    (FunArg a) \n"
  "    (FunArg b) \n"
  "    (Binary + \n"
  "      (Var a) \n"
  "      (Var b)))");
  test("fn sum(i64 a, i64 b) return a+b",
  "  (FunDef sum \n"
  "    (FunArg a) \n"
  "    (FunArg b) \n"
  "    (Return \n"
  "      (Binary + \n"
  "        (Var a) \n"
  "        (Var b))))");
  test("fn sum(i64 a, i64 b[]) { a=a+b return a+b }",
  "  (FunDef sum \n"
  "    (FunArg a) \n"
  "    (FunArg b) \n"
  "    (Block \n"
  "      (Assign = \n"
  "        (Var a) \n"
  "        (Binary + \n"
  "          (Var a) \n"
  "          (Var b))) \n"
  "      (Return \n"
  "        (Binary + \n"
  "          (Var a) \n"
  "          (Var b)))))");
}
