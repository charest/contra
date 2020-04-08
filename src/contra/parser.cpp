#include "errors.hpp"
#include "identifier.hpp"
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
std::unique_ptr<NodeAST> Parser::parseIntegerExpr() {
  auto NumVal = std::atoi( TheLex_.getIdentifierStr().c_str() );
  auto Result = std::make_unique<IntegerExprAST>(getCurLoc(), NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// numberexpr ::= number
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseRealExpr() {
  auto NumVal = std::atof( TheLex_.getIdentifierStr().c_str() );
  auto Result = std::make_unique<RealExprAST>(getCurLoc(), NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// stringexpr ::= string
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseStringExpr() {
  auto Result = std::make_unique<StringExprAST>(getCurLoc(), TheLex_.getIdentifierStr());
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseParenExpr() {
  getNextToken(); // eat (.
  auto V = parseExpression();

  if (CurTok_ != ')') { 
    THROW_SYNTAX_ERROR("Expected ')' after expression", getCurLoc());
  }
  getNextToken(); // eat ).
  return V;
}

//==============================================================================
// identifierexpr
//   ::= 
//   ::= identifier '(' expression* ')'
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseIdentifierExpr() {
  std::string IdName = TheLex_.getIdentifierStr();

  auto LitLoc = getCurLoc();

  getNextToken(); // eat identifier.

  //----------------------------------------------------------------------------
  // Simple variable ref.
  if (CurTok_ != '(') {
    
    // array value load
    if (CurTok_ == '[') {
      getNextToken(); // eat [
      auto Arg = parseExpression();
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR( "Expected ']' at the end of array expression", getCurLoc());
      getNextToken(); // eat ]
      return std::make_unique<VariableExprAST>(LitLoc, IdName, std::move(Arg));
    }
    // scalar load
    else {
      return std::make_unique<VariableExprAST>(LitLoc, IdName);
    }

  } // variable reference

  //----------------------------------------------------------------------------
  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<NodeAST>> Args;
  if (CurTok_ != ')') {
    while (true) {
      auto Arg = parseExpression();
      Args.push_back(std::move(Arg));

      if (CurTok_ == ')')
        break;

      if (CurTok_ != ',')
        THROW_SYNTAX_ERROR("Expected ')' or ',' in argument list", getCurLoc());
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
std::unique_ptr<NodeAST> Parser::parseIfExpr() {

  std::list< std::pair<SourceLocation, std::unique_ptr<NodeAST>> >  Conds;
  ASTBlockList BBlocks;
  
  //---------------------------------------------------------------------------
  // If
  {

    auto IfLoc = getCurLoc();
    getNextToken(); // eat the if.

    // condition.
    auto Cond = parseExpression();
    Conds.emplace_back( IfLoc, std::move(Cond) );

    if (CurTok_ != tok_then)
      THROW_SYNTAX_ERROR("Expected 'then' after 'if'", IfLoc);
    getNextToken(); // eat the then

    // make a new block
    auto Then = createBlock(BBlocks);

    // then
    while (CurTok_ != tok_end && CurTok_ != tok_elif && CurTok_ != tok_else) {
      auto E = parseExpression();
      Then->emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (CurTok_ == tok_elif) {
  
    auto ElifLoc = getCurLoc();
    getNextToken(); // eat elif

    // condition.
    auto Cond = parseExpression();
    Conds.emplace_back( ElifLoc, std::move(Cond) );
  
    if (CurTok_ != tok_then)
      THROW_SYNTAX_ERROR("Expected 'then' after 'elif'", ElifLoc);
    getNextToken(); // eat the then
  
    // make a new block
    auto Then = createBlock(BBlocks);

    while (CurTok_ != tok_end && CurTok_ != tok_elif && CurTok_ != tok_else) {
      auto E = parseExpression();
      Then->emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (CurTok_ == tok_else) {

    getNextToken(); // eat else
    
    // make a new block
    auto Else = createBlock(BBlocks);

    while (CurTok_ != tok_end) {
      auto E = parseExpression();
      Else->emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }

  }
   
  getNextToken(); // eat end
  
  //---------------------------------------------------------------------------
  // Construct If Else Then tree

  return IfStmtAST::makeNested( Conds, BBlocks );
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseForExpr() {
  auto ForLoc = getCurLoc();
  
  bool IsForEach = CurTok_ == tok_foreach;

  getNextToken(); // eat the for.

  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after 'for'", ForLoc);
  std::string IdName = TheLex_.getIdentifierStr();
  auto IdentLoc = getCurLoc();
  getNextToken(); // eat identifier.

  if (CurTok_ != tok_in)
    THROW_SYNTAX_ERROR("Expected 'in' after 'for'", getCurLoc());
  getNextToken(); // eat in

  auto StartLoc = getLexLoc();
  auto Start = parseExpression();

  ForStmtAST::LoopType Loop;
  if (CurTok_ == tok_to) {
    Loop = ForStmtAST::LoopType::To;
  }
  else if (CurTok_ == tok_until) {
    Loop = ForStmtAST::LoopType::Until;
  }
  else
    THROW_SYNTAX_ERROR("Expected 'to' after for start value in 'for' loop", StartLoc);
  getNextToken(); // eat to

  auto End = parseExpression();

  // The step value is optional.
  std::unique_ptr<NodeAST> Step;
  if (CurTok_ == tok_by) {
    getNextToken();
    Step = parseExpression();
  }

  if (CurTok_ != tok_do)
    THROW_SYNTAX_ERROR("Expected 'do' after 'for'", getCurLoc());
  getNextToken(); // eat 'do'.
  
  // add statements
  ASTBlock Body;
  while (CurTok_ != tok_end) {
    auto E = parseExpression();
    Body.emplace_back( std::move(E) );
    if (CurTok_ == tok_sep) getNextToken();
  }
  
  // make a for loop
  auto Id = Identifier{IdName, IdentLoc};
  std::unique_ptr<NodeAST> F;
  if (IsForEach)
    F = std::make_unique<ForeachStmtAST>(getCurLoc(), Id, std::move(Start),
      std::move(End), std::move(Step), std::move(Body), Loop);
  else
    F = std::make_unique<ForStmtAST>(getCurLoc(), Id, std::move(Start),
      std::move(End), std::move(Step), std::move(Body), Loop);

  
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
std::unique_ptr<NodeAST> Parser::parsePrimary() {
 
  switch (CurTok_) {
  case tok_identifier:
    return parseIdentifierExpr();
  case tok_real_number:
    return parseRealExpr();
  case tok_int_number:
    return parseIntegerExpr();
  case '(':
    return parseParenExpr();
  case tok_if:
    return parseIfExpr();
  case tok_for:
  case tok_foreach:
    return parseForExpr();
  case tok_var:
    return parseVarDefExpr();
  case tok_string:
    return parseStringExpr();
  default:
    THROW_SYNTAX_ERROR("Unknown token '" <<  Tokens::getName(CurTok_)
        << "' when expecting an expression", getCurLoc());
  }
}

//==============================================================================
// binoprhs
//   ::= ('+' primary)*
//==============================================================================
std::unique_ptr<NodeAST>
Parser::parseBinOpRHS(int ExprPrec, std::unique_ptr<NodeAST> LHS)
{
  
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = getTokPrecedence();
    
    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok_;
    auto BinLoc = getCurLoc();
    getNextToken(); // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = parseUnary();

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = getTokPrecedence();

    if (TokPrec < NextPrec) {
      RHS = parseBinOpRHS(TokPrec + 1, std::move(RHS));
    }

    // Merge LHS/RHS.
    LHS = std::make_unique<BinaryExprAST>(BinLoc, BinOp, std::move(LHS),
        std::move(RHS));
  }

  return nullptr;
}

//==============================================================================
// expression
//   ::= primary binoprhs
//
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseExpression() {
  auto LHS = parseUnary();

  auto RHS = parseBinOpRHS(0, std::move(LHS));
  return RHS;
}

//==============================================================================
// definition ::= 'def' prototype expression
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseDefinition() {

  getNextToken(); // eat def.
  auto Proto = parsePrototype();

  auto E = parseExpression();
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
}

//==============================================================================
// toplevelexpr ::= expression
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseTopLevelExpr() {

  auto FnLoc = getCurLoc();
  auto E = parseExpression();
  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>( Identifier{"__anon_expr", FnLoc} );
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
}

//==============================================================================
// external ::= 'extern' prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parseExtern() {
  getNextToken(); // eat extern.
  return parsePrototype();
}

//==============================================================================
// unary
//   ::= primary
//   ::= '!' unary
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseUnary() {

  // If the current token is not an operator, it must be a primary expr.
  if (!isTokOperator() || CurTok_ == '(' || CurTok_ == ',') {
    auto P = parsePrimary();
    return P;
  }

  // If this is a unary operator, read it.
  int Opc = CurTok_;
  getNextToken();
  auto Operand = parseUnary();
  return std::make_unique<UnaryExprAST>(getCurLoc(), Opc, std::move(Operand));
}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseVarDefExpr() {

  getNextToken();  // eat the var.
  // At least one variable name is required.
  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after var", getCurLoc());

  std::vector<Identifier> VarNames;
  VarNames.emplace_back(TheLex_.getIdentifierStr(), getCurLoc());
  getNextToken();  // eat identifier.

  Identifier VarType;
  bool IsArray = false;

  std::unique_ptr<NodeAST> Size;

  // get additional variables
  while (CurTok_ == ',') {
    getNextToken();  // eat ','  
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Only variable names are allowed in definition.", getCurLoc());
    VarNames.emplace_back( TheLex_.getIdentifierStr(), getCurLoc() );
    getNextToken();  // eat identifier
  }

  // read modifiers
  if (CurTok_ == ':') {
    getNextToken(); // eat the ':'.
    
    if (CurTok_ == '[') {
      IsArray = true;
      getNextToken(); // eat the '['.
    }

    if (CurTok_ == tok_identifier) {
      VarType = Identifier{TheLex_.getIdentifierStr(), getCurLoc()};
      getNextToken(); // eat the identifier
    }

    if (IsArray) {

      if (CurTok_ != ']' && CurTok_ != ';')
        THROW_SYNTAX_ERROR("Array definition expected ']' or ';' instead of '"
            << Tokens::getName(CurTok_) << "'", getCurLoc());
      else if (CurTok_ == ';') {
        getNextToken(); // eat ;
        Size = parseExpression();
      }
      
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR("Array definition must end with ']' instead of '"
            << Tokens::getName(CurTok_) << "'", getCurLoc());

      getNextToken(); // eat [
    }

  }
  
  // Read the optional initializer.
  std::unique_ptr<NodeAST> Init;
  auto EqLoc = getCurLoc();
  if (CurTok_ == tok_asgmt) {
    getNextToken(); // eat the '='.
    
    if (CurTok_ == '[')
      Init = parseArrayExpr();
    else {
      Init = parseExpression();
    }
  }
  else {
    std::vector<std::string> Names;
    for ( auto i : VarNames ) Names.emplace_back(i.getName());
    THROW_SYNTAX_ERROR("Variable definition for '" << Names << "'"
        << " has no initializer", EqLoc);
  }

  if (IsArray)
    return std::make_unique<ArrayDeclAST>(getCurLoc(), VarNames,
        VarType, std::move(Init), std::move(Size));
  else
    return std::make_unique<VarDeclAST>(getCurLoc(), VarNames, VarType,
        std::move(Init));
}


//==============================================================================
// Array expression parser
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseArrayExpr()
{

  auto Loc = getCurLoc();
  getNextToken(); // eat [.

  ASTBlock ValExprs;
  std::unique_ptr<NodeAST> SizeExpr;

  while (CurTok_ != ']') {
    auto E = parseExpression();

    ValExprs.emplace_back( std::move(E) );
    
    if (CurTok_ == ';') {
      getNextToken(); // eat ;
      if (CurTok_ == ']') {
        THROW_SYNTAX_ERROR("Expected size expression after ';'", getCurLoc());
      }
      SizeExpr = std::move(parseExpression());
      break;
    }

    if (CurTok_ == ',') getNextToken();
  }

  if (CurTok_ != ']')
    THROW_SYNTAX_ERROR( "Expected ']'", getCurLoc() );

 
  // eat ]
  getNextToken();

  return std::make_unique<ArrayExprAST>(Loc, std::move(ValExprs),
      std::move(SizeExpr));
}


//==============================================================================
// Toplevel function parser
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseFunction() {

  bool IsTask = (CurTok_ == tok_task);

  getNextToken(); // eat 'function' / 'task'
  auto Proto = parsePrototype();

  ASTBlock Body;
  std::unique_ptr<NodeAST> Return;

  while (CurTok_ != tok_end) {

    if (CurTok_ == tok_return) {
      getNextToken(); // eat return
      Return = parseExpression();
      break;
    }

    auto E = parseExpression();

    Body.emplace_back( std::move(E) );

    if (CurTok_ == tok_sep) getNextToken();
  }

  if (CurTok_ != tok_end)
    THROW_SYNTAX_ERROR( "Only one return statement allowed for a function.",
        getCurLoc() );
  
  // eat end
  getNextToken();
 
  if (IsTask) {
    return std::make_unique<TaskAST>(std::move(Proto), std::move(Body),
      std::move(Return));
  }
  else
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body),
      std::move(Return));
}

//==============================================================================
// prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parsePrototype() {

  std::string FnName;

  auto FnLoc = getCurLoc();

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok_) {
  default:
    THROW_SYNTAX_ERROR("Expected function name in prototype", getCurLoc());
  case tok_identifier:
    FnName = TheLex_.getIdentifierStr();
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR("Expected unary operator", getCurLoc());
    FnName = "unary";
    FnName += (char)CurTok_;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR("Expected binrary operator", getCurLoc());
    FnName = "binary";
    FnName += (char)CurTok_;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok_ == tok_int_number) {
      auto NumVal = std::atoi(TheLex_.getIdentifierStr().c_str());
      if (NumVal < 1 || NumVal > 100)
        THROW_SYNTAX_ERROR("Invalid precedence of '" << NumVal
            << "' must be between 1 and 100", getCurLoc());
      BinaryPrecedence = NumVal;
      getNextToken();
    }
    else {
      THROW_SYNTAX_ERROR("Precedence must be an integer number", getCurLoc());
    }
    break;
  }
  
  if (CurTok_ != '(')
    THROW_SYNTAX_ERROR("Expected '(' in prototype", getCurLoc());

  getNextToken(); // eat "("

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  while (CurTok_ == tok_identifier) {

    bool IsArray = false;

    auto Loc = getCurLoc();
    auto Name = TheLex_.getIdentifierStr();
    Args.emplace_back( Name, Loc );
    getNextToken(); // eat identifier
    
    if (CurTok_ != ':') 
      THROW_SYNTAX_ERROR("Variable '" << Name << "' needs a type specifier", getCurLoc());
    getNextToken(); // eat ":"
    
    if (CurTok_ == '[') {
      IsArray = true;
      getNextToken(); // eat the '['.
    }
    ArgIsArray.push_back( IsArray );
   
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Variable '" << Name << "' requires a type in prototype"
          << " for function '" << FnName << "'", getCurLoc());
    auto VarType = TheLex_.getIdentifierStr();
    ArgTypes.emplace_back( VarType, getCurLoc() );
    getNextToken(); // eat vartype
    
    if (IsArray) {
      if (CurTok_ != ']') 
        THROW_SYNTAX_ERROR("Array declaration expected ']' instead of '"
          << Tokens::getName(CurTok_) << "'", getCurLoc());
      getNextToken(); // eat ]
    }

    if (CurTok_ == ',') getNextToken(); // eat ','
  }

  if (CurTok_ != ')')
    THROW_SYNTAX_ERROR("Expected ')' in prototype", getCurLoc());

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && Args.size() != Kind)
    THROW_SYNTAX_ERROR("Invalid number of operands for operator: "
        << Kind << " expected, but got " << Args.size(), getCurLoc());

  std::unique_ptr<Identifier> ReturnType;
  if (CurTok_ == '-') {
    getNextToken(); // eat -
    if (CurTok_ != '>')
      THROW_SYNTAX_ERROR("Expected '>' after '-' for return statements", getCurLoc());
    getNextToken(); // eat >

    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Return type requires an identifier in prototype"
          << " for function '" << FnName << "'", getCurLoc());
    ReturnType = std::make_unique<Identifier>(TheLex_.getIdentifierStr(), getCurLoc());
    getNextToken(); // eat vartype
  }


  return std::make_unique<PrototypeAST>( Identifier{FnName, FnLoc}, std::move(Args),
      std::move(ArgTypes), std::move(ArgIsArray), std::move(ReturnType), Kind != 0,
      BinaryPrecedence);
}

} // namespace
