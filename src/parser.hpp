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
  Lexer TheLex;

  // verbosity is on
  bool IsVerbose = false;

public:

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int CurTok;

  /// BinopPrecedence - This holds the precedence for each binary operator that is
  /// defined.
  std::map<char, int> BinopPrecedence;
  
  Parser(bool IsVerbose = false) : IsVerbose(IsVerbose) {
    setBinopPrecedence();
  }

  Parser( const std::string & filename, bool IsVerbose = false )
    : TheLex(filename), IsVerbose(IsVerbose)
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

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int getNextToken() {
    CurTok = TheLex.gettok();
    //if ( CurTok == tok_identifier )
    //  std::cerr << "echo> " << getTokName(CurTok) << " value " << TheLex.IdentifierStr << std::endl;
    //else
    //  std::cerr << "echo> " << getTokName(CurTok) << std::endl;
    return CurTok;
  }

  /// GetTokPrecedence - Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(CurTok))
      return -1;

    // Make sure it's a declared binop.
    auto TokPrecIt = BinopPrecedence.find(CurTok);
    if (TokPrecIt == BinopPrecedence.end()) return -1;
    return TokPrecIt->second;
  }

  int getLine() { return TheLex.CurLoc.Line; }
  
  void echo(const std::string & msg, int Depth) {
    if (IsVerbose) {
      std::cerr << std::string(2*Depth, '.');
      std::cerr << msg << std::endl;
    }
  }

  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseIntegerExpr(int Depth = 0);
  std::unique_ptr<ExprAST> parseRealExpr(int Depth = 0);
  
  /// stringexpr ::= string
  std::unique_ptr<ExprAST> parseStringExpr(int Depth = 0);

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr(int Depth = 0);
  
  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression* ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr(int Depth = 0);
  
  /// ifexpr ::= 'if' expression 'then' expression 'else' expression
  std::unique_ptr<ExprAST> parseIfExpr(int Depth = 0);
  
  /// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
  std::unique_ptr<ExprAST> parseForExpr(int Depth = 0);
  
  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= ifexpr
  ///   ::= forexpr
  std::unique_ptr<ExprAST> parsePrimary(int Depth = 0);
  
  /// binoprhs
  ///   ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS, int Depth = 0);
  
  /// expression
  ///   ::= primary binoprhs
  ///
  std::unique_ptr<ExprAST> parseExpression(int Depth = 0);

  /// definition ::= 'def' prototype expression
  std::unique_ptr<FunctionAST> parseDefinition(int Depth = 0);
  

  /// toplevelexpr ::= expression
  std::unique_ptr<FunctionAST> parseTopLevelExpr(int Depth = 0);
  
  /// external ::= 'extern' prototype
  std::unique_ptr<PrototypeAST> parseExtern(int Depth = 0);

  /// unary
  ///   ::= primary
  ///   ::= '!' unary
  std::unique_ptr<ExprAST> parseUnary(int Depth = 0);

  /// varexpr ::= 'var' identifier ('=' expression)?
  ///                    (',' identifier ('=' expression)?)* 'in' expression
  std::unique_ptr<ExprAST> parseVarExpr(int Depth = 0);

  /// Top level function parser 
  std::unique_ptr<FunctionAST> parseFunction(int Depth = 0);
  
  /// prototype
  std::unique_ptr<PrototypeAST> parsePrototype(int Depth = 0);
};

} // namespace

#endif // PARSER_HPP
