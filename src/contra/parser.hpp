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
  int CurTok_ = tok_eof;
  int NextTok_ = tok_eof;

  SourceLocation CurLoc_;
  LocationRange IdentifierLoc_;
  std::string IdentifierStr_;

  /// BinopPrecedence - This holds the precedence for each binary operator that is
  /// defined.
  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

public:

  Parser(std::shared_ptr<BinopPrecedence> Precedence) :
    BinopPrecedence_(Precedence)
  { getNextToken(); }

  Parser(std::shared_ptr<BinopPrecedence> Precedence,
      const std::string & filename ) :
    TheLex_(filename), BinopPrecedence_(Precedence)
  { getNextToken(); }

  /// get the current token
  int getCurTok() const { return CurTok_; }

  /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
  /// token the parser is looking at.  getNextToken reads another token from the
  /// lexer and updates CurTok with its results.
  int getNextToken() {
    // save state
    CurTok_ = NextTok_;
    CurLoc_ = TheLex_.getCurLoc();
    IdentifierLoc_ = TheLex_.getIdentifierLoc();
    IdentifierStr_ = TheLex_.getIdentifierStr();
    // advance
    NextTok_ = TheLex_.gettok();
    return CurTok_;
  }

  /// GetTokPrecedence - Get the precedence of the pending binary operator token.
  int getTokPrecedence() const {
    // Make sure it's a declared binop.
    auto TokPrecIt = BinopPrecedence_->find(CurTok_);
    if (TokPrecIt.found) 
      return TokPrecIt.precedence;
    else
      return -1;
  }

  bool isTokOperator() const
  { return BinopPrecedence_->count(CurTok_); }

  const auto & getCurLoc() const { return CurLoc_; }
  const auto & getIdentifierLoc() const { return IdentifierLoc_; }
  const auto & getIdentifierStr() const { return IdentifierStr_; }
  
  // print out current line
  std::ostream & barf(std::ostream& out, const LocationRange & Loc)
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
  std::unique_ptr<NodeAST> parseRangeExpr();
  
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

  /// toplevelexpr ::= expression
  std::unique_ptr<FunctionAST> parseTopLevelExpr();
  
  /// unary
  ///   ::= primary
  ///   ::= '!' unary
  std::unique_ptr<NodeAST> parseUnary();
  
  std::unique_ptr<NodeAST> parseStatement();

  /// varexpr ::= 'var' identifier ('=' expression)?
  ///                    (',' identifier ('=' expression)?)* 'in' expression
  std::unique_ptr<NodeAST> parseVarDefExpr();
  
  std::unique_ptr<NodeAST> parsePartitionExpr();

  /// Top level function parser 
  std::unique_ptr<FunctionAST> parseFunction();
  
  /// prototype
  std::unique_ptr<PrototypeAST> parsePrototype();
};

} // namespace

#endif // PARSER_HPP
