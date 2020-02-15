#ifndef PARSER_HPP
#define PARSER_HPP

#include "ast.hpp"
#include "lexer.hpp"
#include "token.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace contra {

class Parser {

  // A lexer object
  Lexer L;

public:

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int CurTok;

  /// BinopPrecedence - This holds the precedence for each binary operator that is
  /// defined.
  std::map<char, int> BinopPrecedence;
  
  Parser() {
    // Install standard binary operators.
    // 1 is lowest precedence.
    BinopPrecedence[op_lt] = 10;
    BinopPrecedence[op_add] = 20;
    BinopPrecedence[op_sub] = 20;
    BinopPrecedence[op_mul] = 40;
    BinopPrecedence[op_div] = 50;
    // highest.
  }

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int getNextToken() { return CurTok = L.gettok(); }

  /// GetTokPrecedence - Get the precedence of the pending binary operator token.
  int GetTokPrecedence() {
    if (!isascii(CurTok))
      return -1;

    // Make sure it's a declared binop.
    int TokPrec = BinopPrecedence[CurTok];
    if (TokPrec <= 0) return -1;
    return TokPrec;
  }
  
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> ParseNumberExpr();
  
  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> ParseParenExpr();
  
  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression* ')'
  std::unique_ptr<ExprAST> ParseIdentifierExpr();
  
  /// ifexpr ::= 'if' expression 'then' expression 'else' expression
  std::unique_ptr<ExprAST> ParseIfExpr();
  
  /// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
  std::unique_ptr<ExprAST> ParseForExpr();
  
  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= ifexpr
  ///   ::= forexpr
  std::unique_ptr<ExprAST> ParsePrimary();
  
  /// binoprhs
  ///   ::= ('+' primary)*
  std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS);
  
  /// expression
  ///   ::= primary binoprhs
  ///
  std::unique_ptr<ExprAST> ParseExpression();
  
  /// prototype
  ///   ::= id '(' id* ')'
  std::unique_ptr<PrototypeAST> ParsePrototype();
  
  /// definition ::= 'def' prototype expression
  std::unique_ptr<FunctionAST> ParseDefinition();
  
  /// toplevelexpr ::= expression
  std::unique_ptr<FunctionAST> ParseTopLevelExpr();
  
  /// external ::= 'extern' prototype
  std::unique_ptr<PrototypeAST> ParseExtern();

};

} // namespace

#endif // PARSER_HPP
