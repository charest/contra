#ifndef PARSER_HPP
#define PARSER_HPP

#include "ast.hpp"
#include "lexer.hpp"
#include "precedence.hpp"
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

  /// BinopPrecedence - This holds the precedence for each binary operator that is
  /// defined.
  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

public:

  Parser(std::shared_ptr<BinopPrecedence> Precedence) :
    BinopPrecedence_(Precedence)
  {}

  Parser(std::shared_ptr<BinopPrecedence> Precedence,
      const std::string & filename ) :
    TheLex_(filename), BinopPrecedence_(Precedence)
  {}

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
  int getTokPrecedence() const {
    if (!isascii(CurTok_))
      return -1;

    // Make sure it's a declared binop.
    auto TokPrecIt = BinopPrecedence_->find(CurTok_);
    if (TokPrecIt.found) 
      return TokPrecIt.precedence;
    else
      return -1;
  }

  const auto & getCurLoc() const { return TheLex_.getCurLoc(); }
  const auto & getLexLoc() const { return TheLex_.getLexLoc(); }
  
  // print out current line
  std::ostream & barf(std::ostream& out, const SourceLocation & Loc)
  { return TheLex_.barf(out, Loc); } 

  std::shared_ptr<BinopPrecedence> getBinopPrecedence() const
  { return BinopPrecedence_; }

  /// numberexpr ::= number
  std::unique_ptr<NodeAST> parseIntegerExpr();
  std::unique_ptr<NodeAST> parseRealExpr();
  
  /// stringexpr ::= string
  std::unique_ptr<NodeAST> parseStringExpr();

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<NodeAST> parseParenExpr();
  
  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression* ')'
  std::unique_ptr<NodeAST> parseIdentifierExpr();
  
  /// ifexpr ::= 'if' expression 'then' expression 'else' expression
  std::unique_ptr<NodeAST> parseIfExpr();
  
  /// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
  std::unique_ptr<NodeAST> parseForExpr();

  std::unique_ptr<NodeAST> parseArrayExpr();
  
  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= ifexpr
  ///   ::= forexpr
  std::unique_ptr<NodeAST> parsePrimary();
  
  /// binoprhs
  ///   ::= ('+' primary)*
  std::unique_ptr<NodeAST> parseBinOpRHS(int ExprPrec, std::unique_ptr<NodeAST> LHS);
  
  /// expression
  ///   ::= primary binoprhs
  ///
  std::unique_ptr<NodeAST> parseExpression();

  /// definition ::= 'def' prototype expression
  std::unique_ptr<FunctionAST> parseDefinition();
  

  /// toplevelexpr ::= expression
  std::unique_ptr<FunctionAST> parseTopLevelExpr();
  
  /// external ::= 'extern' prototype
  std::unique_ptr<PrototypeAST> parseExtern();

  /// unary
  ///   ::= primary
  ///   ::= '!' unary
  std::unique_ptr<NodeAST> parseUnary();

  /// varexpr ::= 'var' identifier ('=' expression)?
  ///                    (',' identifier ('=' expression)?)* 'in' expression
  std::unique_ptr<NodeAST> parseVarDefExpr();

  /// Top level function parser 
  std::unique_ptr<FunctionAST> parseFunction();
  
  /// prototype
  std::unique_ptr<PrototypeAST> parsePrototype();
};

} // namespace

#endif // PARSER_HPP
