#include "errors.hpp"
#include "parser.hpp"
#include "string_utils.hpp"

#include <iomanip>
#include <list>
#include <utility>
#include <vector>

namespace contra {

  
//==============================================================================
// numberexpr ::= number
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseNumberExpr(int Depth) {
  echo( Formatter() << "Parsing number expression '" << TheLex.NumVal << "'", Depth++ );
  auto Result = std::make_unique<NumberExprAST>(TheLex.CurLoc, TheLex.NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// stringexpr ::= string
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseStringExpr(int Depth) {
  echo( Formatter() << "Parsing string expression '"
      << escape(TheLex.IdentifierStr) << "'", Depth++ );
  auto Result = std::make_unique<StringExprAST>(TheLex.CurLoc, TheLex.IdentifierStr);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseParenExpr(int Depth) {
  echo( "Parsing parenthases expression", Depth++ ); 
  getNextToken(); // eat (.
  auto V = parseExpression(++Depth);

  if (CurTok != ')')
    THROW_SYNTAX_ERROR("Expected ')'", getLine());
  getNextToken(); // eat ).
  return V;
}

//==============================================================================
// identifierexpr
//   ::= 
//   ::= identifier '(' expression* ')'
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseIdentifierExpr(int Depth) {
  std::string IdName = TheLex.IdentifierStr;
  echo( Formatter() << "Parsing identifyer expression '" << IdName << "'", Depth++ ); 

  SourceLocation LitLoc = TheLex.CurLoc;

  getNextToken(); // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return std::make_unique<VariableExprAST>(LitLoc, IdName);

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      auto Arg = parseExpression(Depth);
      Args.push_back(std::move(Arg));

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        THROW_SYNTAX_ERROR("Expected ')' or ',' in argument list", getLine());
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(LitLoc, IdName, std::move(Args));
}

//==============================================================================
// ifexpr ::= 'if' expression 'then' expression 'else' expression
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseIfExpr(int Depth) {
  echo( "Parsing conditional expression", Depth++ ); 

  using block_t = std::vector< std::unique_ptr<ExprAST> >;
  std::list< std::pair< SourceLocation, std::unique_ptr<ExprAST> > > Conds;
  std::list< block_t > BBlocks;
  
  //---------------------------------------------------------------------------
  // If
  {

    auto IfLoc = TheLex.CurLoc;
    getNextToken(); // eat the if.

    // condition.
    auto Cond = parseExpression(Depth);
    Conds.emplace_back( std::make_pair(IfLoc, std::move(Cond)) );

    if (CurTok != tok_then)
      THROW_SYNTAX_ERROR("Expected 'then' after 'if'", getLine());
    getNextToken(); // eat the then

    // make a new block
    auto Then = BBlocks.emplace( BBlocks.end(), block_t{} );

    // then
    while (CurTok != tok_end && CurTok != tok_elif && CurTok != tok_else) {
      auto E = parseExpression(Depth);
      Then->emplace_back( std::move(E) );
      if (CurTok == tok_sep) getNextToken();
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (CurTok == tok_elif) {
  
    auto ElifLoc = TheLex.CurLoc;
    getNextToken(); // eat elif

    // condition.
    auto Cond = parseExpression(Depth);
    Conds.emplace_back( std::make_pair(ElifLoc, std::move(Cond)) );
  
    if (CurTok != tok_then)
      THROW_SYNTAX_ERROR("Expected 'then' after 'elif'", getLine());
    getNextToken(); // eat the then
  
    // make a new block
    auto Then = BBlocks.emplace( BBlocks.end(), block_t{} );

    while (CurTok != tok_end && CurTok != tok_elif && CurTok != tok_else) {
      auto E = parseExpression(Depth);
      Then->emplace_back( std::move(E) );
      if (CurTok == tok_sep) getNextToken();
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (CurTok == tok_else) {

    auto ElseLoc = TheLex.CurLoc;
    getNextToken(); // eat else
    
    // make a new block
    auto Else = BBlocks.emplace( BBlocks.end(), block_t{} );

    while (CurTok != tok_end) {
      auto E = parseExpression(Depth);
      Else->emplace_back( std::move(E) );
      if (CurTok == tok_sep) getNextToken();
    }

  }
   
  getNextToken(); // eat end
  
  //---------------------------------------------------------------------------
  // Construct If Else Then tree

  return IfExprAST::make( Conds, BBlocks );
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseForExpr(int Depth) {
  echo( "Parsing for expression", Depth++ ); 
  getNextToken(); // eat the for.

  if (CurTok != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after 'for'", getLine());
  std::string IdName = TheLex.IdentifierStr;
  getNextToken(); // eat identifier.
  
  if (CurTok != tok_in)
    THROW_SYNTAX_ERROR("Expected 'in' after 'for'", getLine());
  getNextToken(); // eat in

  auto Start = parseExpression(Depth);

  if (CurTok != tok_to)
    THROW_SYNTAX_ERROR("Expected 'to' after for start value in 'for' loop", getLine());
  getNextToken(); // eat to

  auto End = parseExpression(Depth);

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == tok_by) {
    getNextToken();
    Step = parseExpression(Depth);
  }

  if (CurTok != tok_do)
    THROW_SYNTAX_ERROR("Expected 'do' after 'for'", getLine());
  getNextToken(); // eat 'do'.
  
  // make a for loop
  auto F = std::make_unique<ForExprAST>(TheLex.CurLoc, IdName, std::move(Start),
      std::move(End), std::move(Step));

  // add statements
  while (CurTok != tok_end) {
    auto E = parseExpression(Depth);
    F->Body.emplace_back( std::move(E) );
    if (CurTok == tok_sep) getNextToken();
  }
  
  // eat end
  getNextToken();

  return F;
}

//==============================================================================
// primary
//   ::= identifierexpr
//   ::= numberexpr
//   ::= parenexpr
//   ::= ifexpr
//   ::= forexpr
//   ::= varexpr
//==============================================================================
std::unique_ptr<ExprAST> Parser::parsePrimary(int Depth) {
  echo( "Parsing primary expression", Depth++ ); 
  
  switch (CurTok) {
  case tok_identifier:
    return parseIdentifierExpr(Depth);
  case tok_number:
    return parseNumberExpr(Depth);
  case '(':
    return parseParenExpr(Depth);
  case tok_if:
    return parseIfExpr(Depth);
  case tok_for:
    return parseForExpr(Depth);
  case tok_var:
    return parseVarExpr(Depth);
  case tok_end:
  case tok_return:
    std::cerr << "HERHERHEHER "<< getTokName(CurTok) << std::endl;
    abort();
    //return ParseReturnExpr();
  case tok_string:
    return parseStringExpr(Depth);
  default:
    THROW_SYNTAX_ERROR("Unknown token '" <<  getTokName(CurTok)
        << "' when expecting an expression", getLine());
  }
}

//==============================================================================
// binoprhs
//   ::= ('+' primary)*
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseBinOpRHS(int ExprPrec,
    std::unique_ptr<ExprAST> LHS, int Depth) {

  echo( "Parsing binary expression", Depth++ ); 
  
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = getTokPrecedence();
    
    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    SourceLocation BinLoc = TheLex.CurLoc;
    getNextToken(); // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = parseUnary(Depth);

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = getTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = parseBinOpRHS(TokPrec + 1, std::move(RHS), Depth);
    }

    // Merge LHS/RHS.
    LHS = std::make_unique<BinaryExprAST>(BinLoc, BinOp, std::move(LHS),
        std::move(RHS));
  }
}

//==============================================================================
// expression
//   ::= primary binoprhs
//
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseExpression(int Depth) {
  echo( "Parsing general expression", Depth++ ); 
  
  auto LHS = parseUnary(Depth);

  auto RHS = parseBinOpRHS(0, std::move(LHS), Depth);
  return std::move(RHS);
}

//==============================================================================
// definition ::= 'def' prototype expression
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseDefinition(int Depth) {
  echo( "Parsing definition", Depth++ ); 

  getNextToken(); // eat def.
  auto Proto = parsePrototype(Depth);

  auto E = parseExpression(Depth);
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
}

//==============================================================================
// toplevelexpr ::= expression
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseTopLevelExpr(int Depth) {
  echo( "Parsing top level expression", Depth++ ); 

  SourceLocation FnLoc = TheLex.CurLoc;
  auto E = parseExpression(Depth);
  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>(FnLoc, 
      "__anon_expr", std::vector<std::string>());
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
}

//==============================================================================
// external ::= 'extern' prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parseExtern(int Depth) {
  echo( "Parsing extern expression", Depth++ ); 
  getNextToken(); // eat extern.
  return parsePrototype(Depth);
}

//==============================================================================
// unary
//   ::= primary
//   ::= '!' unary
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseUnary(int Depth) {
  echo( "Parsing unary expression", Depth++ );

  // If the current token is not an operator, it must be a primary expr.
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',') {
    auto P = parsePrimary(Depth);
    return std::move(P);
  }

  // If this is a unary operator, read it.
  int Opc = CurTok;
  getNextToken();
  auto Operand = parseUnary(Depth);
  return std::make_unique<UnaryExprAST>(TheLex.CurLoc, Opc, std::move(Operand));
}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseVarExpr(int Depth) {
  echo( "Parsing variable expression", Depth++ );

  getNextToken();  // eat the var.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after var", getLine());

  while (1) {
    std::string Name = TheLex.IdentifierStr;
    getNextToken();  // eat identifier.
  
    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.
  
      Init = parseExpression(Depth);
    }
    else {
      THROW_SYNTAX_ERROR("Variable definition for '" << Name << "'"
          << " has no initializer", getLine());
    }
  
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
  
    // End of var list, exit loop.
    if (CurTok != ',') break;
    getNextToken(); // eat the ','.
  
    if (CurTok != tok_identifier)
      THROW_SYNTAX_ERROR("Expected identifier list after var", getLine());
  }
  

  return std::make_unique<VarExprAST>(TheLex.CurLoc, std::move(VarNames));
}



//==============================================================================
// Toplevel function parser
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseFunction(int Depth) {
  echo( "Parsing function", Depth++ );

  getNextToken(); // eat def.
  auto Proto = parsePrototype(Depth);
  
  auto F = std::make_unique<FunctionAST>(std::move(Proto));

  while (CurTok != tok_end) {
    auto E = parseExpression(Depth);

    F->Body.emplace_back( std::move(E) );

    if (CurTok == tok_sep) getNextToken();
  }
  
  // eat end
  getNextToken();

  return std::move(F);
}

//==============================================================================
// prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parsePrototype(int Depth) {
  echo( "Parsing function prototype", Depth++ );

  std::string FnName;

  SourceLocation FnLoc = TheLex.CurLoc;

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    THROW_SYNTAX_ERROR("Expected function name in prototype", getLine());
  case tok_identifier:
    FnName = TheLex.IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      THROW_SYNTAX_ERROR("Expected unary operator", getLine());
    FnName = "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      THROW_SYNTAX_ERROR("Expected binrary operator", getLine());
    FnName = "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (TheLex.NumVal < 1 || TheLex.NumVal > 100)
        THROW_SYNTAX_ERROR("Invalid precedence of '" << TheLex.NumVal
            << "' must be between 1 and 100", getLine());
      BinaryPrecedence = (unsigned) TheLex.NumVal;
      getNextToken();
    }
    break;
  }
  
  if (CurTok != '(')
    THROW_SYNTAX_ERROR("Expected '(' in prototype", getLine());

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(TheLex.IdentifierStr);
  if (CurTok != ')')
    THROW_SYNTAX_ERROR("Expected ')' in prototype", getLine());

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    THROW_SYNTAX_ERROR("Invalid number of operands for operator: "
        << Kind << " expected, but got " << ArgNames.size(), getLine());

  return std::make_unique<PrototypeAST>(FnLoc, FnName,
      std::move(ArgNames), Kind != 0, BinaryPrecedence);
}

} // namespace
