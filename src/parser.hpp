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
  Lexer TheLex_;

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int CurTok_;


public:

  // A symbol table to infer types in the parser
  std::map<std::string, VarTypes> NamedValues;

  /// BinopPrecedence - This holds the precedence for each binary operator that is
  /// defined.
  std::map<char, int> BinopPrecedence;
  
  Parser() {
    setBinopPrecedence();
  }

  Parser( const std::string & filename ) : TheLex_(filename)
  {
    setBinopPrecedence();
  }

  void setBinopPrecedence() {
    // Install standard binary operators.
    // 1 is lowest precedence.
    BinopPrecedence[tok_eq] = 2;
    BinopPrecedence[tok_lt] = 10;
    BinopPrecedence[tok_add] = 20;
    BinopPrecedence[tok_sub] = 20;
    BinopPrecedence[tok_mul] = 40;
    BinopPrecedence[tok_div] = 50;
    // highest.
  }

  /// get the current token
  int getCurTok() const { return CurTok_; }

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int getNextToken() {
    CurTok_ = TheLex_.gettok();
    return CurTok_;
  }

  /// GetTokPrecedence - Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(CurTok_))
      return -1;

    // Make sure it's a declared binop.
    auto TokPrecIt = BinopPrecedence.find(CurTok_);
    if (TokPrecIt == BinopPrecedence.end()) return -1;
    return TokPrecIt->second;
  }

  int getLine() { return TheLex_.getLexLoc().getLine(); }

  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseIntegerExpr();
  std::unique_ptr<ExprAST> parseRealExpr();
  
  /// stringexpr ::= string
  std::unique_ptr<ExprAST> parseStringExpr();

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr();
  
  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression* ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr();
  
  /// ifexpr ::= 'if' expression 'then' expression 'else' expression
  std::unique_ptr<ExprAST> parseIfExpr();
  
  /// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
  std::unique_ptr<ExprAST> parseForExpr();

  std::unique_ptr<ExprAST> parseArrayExpr(VarTypes VarType);
  
  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= ifexpr
  ///   ::= forexpr
  std::unique_ptr<ExprAST> parsePrimary();
  
  /// binoprhs
  ///   ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS);
  
  /// expression
  ///   ::= primary binoprhs
  ///
  std::unique_ptr<ExprAST> parseExpression();

  /// definition ::= 'def' prototype expression
  std::unique_ptr<FunctionAST> parseDefinition();
  

  /// toplevelexpr ::= expression
  std::unique_ptr<FunctionAST> parseTopLevelExpr();
  
  /// external ::= 'extern' prototype
  std::unique_ptr<PrototypeAST> parseExtern();

  /// unary
  ///   ::= primary
  ///   ::= '!' unary
  std::unique_ptr<ExprAST> parseUnary();

  /// varexpr ::= 'var' identifier ('=' expression)?
  ///                    (',' identifier ('=' expression)?)* 'in' expression
  std::unique_ptr<ExprAST> parseVarExpr();

  /// Top level function parser 
  std::unique_ptr<FunctionAST> parseFunction();
  
  /// prototype
  std::unique_ptr<PrototypeAST> parsePrototype();
};

} // namespace

#endif // PARSER_HPP
