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
  auto Result = std::make_unique<ValueExprAST>(
      getIdentifierLoc(),
      getIdentifierStr(),
      ValueExprAST::ValueType::Int);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// numberexpr ::= number
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseRealExpr() {
  auto Result = std::make_unique<ValueExprAST>(
      getIdentifierLoc(),
      getIdentifierStr(),
      ValueExprAST::ValueType::Real);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// stringexpr ::= string
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseStringExpr() {
  auto Result = std::make_unique<ValueExprAST>(
      getIdentifierLoc(),
      getIdentifierStr(),
      ValueExprAST::ValueType::String);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseParenExpr() {
  auto BeginLoc = getCurLoc();
  getNextToken(); // eat (.
  auto V = parseExpression();

  if (CurTok_ != ')') {
    THROW_SYNTAX_ERROR(
        "Expected ')' after expression", 
        getLocationRange(BeginLoc) );
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
  
  auto BeginLoc = getCurLoc();
  auto Id = getIdentifier();

  getNextToken(); // eat identifier.
  
  //----------------------------------------------------------------------------
  // Call.
  if (CurTok_ == '(') {
    auto ArgsBeginLoc = getCurLoc();
    getNextToken(); // eat (
    std::unique_ptr<NodeAST> Args;
    if (CurTok_ != ')') Args = parseExpression();
    if (CurTok_ != ')')
      THROW_NAME_ERROR("Expected ')'.", getLocationRange(ArgsBeginLoc));
    getNextToken(); // Eat the ')'.
    return std::make_unique<CallExprAST>(
        getLocationRange(BeginLoc),
        Id,
        std::move(Args));
  }

  //----------------------------------------------------------------------------
  // Variable reference
  else {
  
    // Has a type, so we know its a decl
    std::unique_ptr<Identifier> VarTypeId;
    if (CurTok_ == tok_identifier && isType(Id.getName())) {
      VarTypeId = std::make_unique<Identifier>(Id);
      getNextToken();  // eat the type
      Id = getIdentifier();
    }

    
    //----------------------------------
    // Array
    if (CurTok_ == '[') {
      auto ArrayLoc = getCurLoc();
      getNextToken(); // eat [
      auto IndexExpr = parseExpression();
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR(
            "Expected ']'  in array access/declaration.",
            getLocationRange(ArrayLoc));
      getNextToken(); // eat ]
      return std::make_unique<ArrayAccessExprAST>(
          getLocationRange(BeginLoc),
          Id,
          std::move(IndexExpr),
          std::move(VarTypeId));
    }
    
    //----------------------------------
    // scalar
    else {
      return std::make_unique<VarAccessExprAST>(
          getLocationRange(BeginLoc),
          Id,
          std::move(VarTypeId));
    }

  } // variable reference

}

//==============================================================================
// ifexpr ::= 'if' expression 'then' expression 'else' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseIfExpr() {

  IfStmtAST::ConditionList Conds;
  ASTBlockList BBlocks;
  
  //---------------------------------------------------------------------------
  // If
  {

    auto IfLoc = getIdentifierLoc();
    getNextToken(); // eat the if.

    // condition.
    auto Cond = parseExpression();
    Conds.emplace_back( IfLoc, std::move(Cond) );
      
    // make a new block
    auto Then = createBlock(BBlocks);
  
    //------------------------------------
    // Multi-liner
    if (CurTok_ == '{') {
      getNextToken(); // eat {
      while (CurTok_ != '}') {
        auto E = parseExpression();
        Then->emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      auto E = parseExpression();
      Then->emplace_back( std::move(E) );
    }

  }
  
  //---------------------------------------------------------------------------
  // Else if

  while (CurTok_ == tok_elif) {
  
    auto ElifLoc = getIdentifierLoc();
    getNextToken(); // eat elif

    // condition.
    auto Cond = parseExpression();
    Conds.emplace_back( ElifLoc, std::move(Cond) );
  
    // make a new block
    auto Then = createBlock(BBlocks);

    //------------------------------------
    // Multi-liner
    if (CurTok_ == '{') {
      getNextToken(); // eat {
      while (CurTok_ != '}') {
        auto E = parseExpression();
        Then->emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      auto E = parseExpression();
      Then->emplace_back( std::move(E) );
    }

  }


  //---------------------------------------------------------------------------
  // Else

  if (CurTok_ == tok_else) {

    getNextToken(); // eat else
    
    // make a new block
    auto Else = createBlock(BBlocks);

    //------------------------------------
    // Multi-liner
    if (CurTok_ == '{') {
      getNextToken(); // eat {
      while (CurTok_ != '}') {
        auto E = parseExpression();
        Else->emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat }
    }
    //------------------------------------
    // One-liner
    else {
      auto E = parseExpression();
      Else->emplace_back( std::move(E) );
    }

  }
  
  //---------------------------------------------------------------------------
  // Construct If Else Then tree

  return IfStmtAST::makeNested( Conds, BBlocks );
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseForExpr() {
  auto BeginLoc = getCurLoc();
  auto ForLoc = getIdentifierLoc();
  
  bool IsForEach = CurTok_ == tok_foreach;

  getNextToken(); // eat the for.

  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after 'for'", getIdentifierLoc());
  std::string IdName = getIdentifierStr();
  auto IdentLoc = getIdentifierLoc();
  getNextToken(); // eat identifier.

  if (CurTok_ != tok_asgmt)
    THROW_SYNTAX_ERROR(
        "Expected '=' after 'for'",
        getLocationRange(BeginLoc));
  getNextToken(); // eat =

  auto Start = parseExpression();

  // add statements
  ASTBlock Body;

  //------------------------------------
  // Multi-liner
  if (CurTok_ == '{') {
    getNextToken(); // eat {
    while (CurTok_ != '}') {
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }
    getNextToken(); // eat }
  }
  //------------------------------------
  // One-liner
  else {
    auto E = parseExpression();
    Body.emplace_back( std::move(E) );
  }
  
  // make a for loop
  auto Id = Identifier{IdName, IdentLoc};
  std::unique_ptr<NodeAST> F;
  if (IsForEach)
    F = std::make_unique<ForeachStmtAST>(
        ForLoc,
        Id,
        std::move(Start),
      std::move(Body));
  else
    F = std::make_unique<ForStmtAST>(
        ForLoc,
        Id,
        std::move(Start),
        std::move(Body));


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
  case tok_real_literal:
    return parseRealExpr();
  case tok_int_literal:
    return parseIntegerExpr();
  case '(':
    return parseParenExpr();
  case '[':
    return parseArrayExpr();
  case tok_if:
    return parseIfExpr();
  case tok_for:
  case tok_foreach:
    return parseForExpr();
  case tok_use:
    return parsePartitionExpr();
  case tok_reduce:
    return parseReductionExpr();
  case tok_string_literal:
    return parseStringExpr();
  default:
    THROW_SYNTAX_ERROR("Unknown token '" <<  Tokens::getName(CurTok_)
        << "' when expecting an expression", getIdentifierLoc());
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
    auto BinLoc = getIdentifierLoc();
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
  auto BeginLoc = getCurLoc();
  
  std::unique_ptr<NodeAST> LHS = parseUnary();
  LHS = parseBinOpRHS(0, std::move(LHS));

  if (CurTok_ == ',') {
    ASTBlock Exprs;
    Exprs.emplace_back( std::move(LHS) );
    while (CurTok_ == ',') {
      getNextToken(); // eat ,
      std::unique_ptr<NodeAST> LHS = parseUnary();
      LHS = parseBinOpRHS(0, std::move(LHS));
      Exprs.emplace_back( std::move(LHS) );
    }
    LHS = std::make_unique<ExprListAST>(
        getLocationRange(BeginLoc),
        std::move(Exprs));
  }
  else if (CurTok_ == ':') {
    ASTBlock Exprs;
    Exprs.emplace_back( std::move(LHS) );
    while (CurTok_ == ':') {
      getNextToken(); // eat :
      std::unique_ptr<NodeAST> LHS = parseUnary();
      LHS = parseBinOpRHS(0, std::move(LHS));
      Exprs.emplace_back( std::move(LHS) );
    }
    if (Exprs.size() > 3 || Exprs.size() < 2)
      THROW_SYNTAX_ERROR(
          "Only 'begin':'end':['step'] specification supported for ranges." ,
          getLocationRange(BeginLoc));
    LHS = std::make_unique<RangeExprAST>(
        getLocationRange(BeginLoc),
        std::move(Exprs));
  }

  if (CurTok_ == tok_asgmt) {
    getNextToken(); // eat =
    auto RHS = parseExpression();
    LHS = std::make_unique<AssignStmtAST>(
        getLocationRange(BeginLoc),
        std::move(LHS),
        std::move(RHS));
  }

  return LHS;
}

//==============================================================================
// toplevelexpr ::= expression
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseTopLevelExpr() {

  auto FnLoc = getIdentifierLoc();
  auto E = parseExpression();
  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>( Identifier{"__anon_expr", FnLoc} );
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
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
  return std::make_unique<UnaryExprAST>(
      getIdentifierLoc(),
      Opc,
      std::move(Operand));
}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parsePartitionExpr() {

  auto BeginLoc = getCurLoc();
  getNextToken();  // eat the use
    
  std::vector<Identifier> RangeIds;

  while (CurTok_ != ':') {
    auto RangeLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'use'.", RangeLoc);
    RangeIds.emplace_back( getIdentifierStr(), RangeLoc );
    getNextToken(); // eat identifier.
    if (CurTok_ == ',') getNextToken(); // eat ,
  }

  auto ColonLoc = getCurLoc();
  if (CurTok_ != ':')
    THROW_SYNTAX_ERROR(
        "Expected ':'.",
        getIdentifierLoc());
  getNextToken(); // eat ":".

  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR(
        "Expected identifier after ':'.",
        getLocationRange(ColonLoc));
  auto PartExpr = parseExpression(); 

  return std::make_unique<PartitionStmtAST>(
      getLocationRange(BeginLoc),
      RangeIds,
      std::move(PartExpr));
}

//==============================================================================
// reduction
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseReductionExpr() {

  auto BeginLoc = getCurLoc();
  getNextToken();  // eat the reduce
    
  std::vector<Identifier> VarIds;

  while (CurTok_ != ':') {
    auto VarLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'reduce'.", VarLoc);
    VarIds.emplace_back( getIdentifierStr(), VarLoc );
    getNextToken(); // eat identifier.
    if (CurTok_ == ',') getNextToken(); // eat ,
  }

  if (CurTok_ != ':')
    THROW_SYNTAX_ERROR(
        "Expected ':'.",
        getIdentifierLoc());
  getNextToken(); // eat ":".

  if (!isTokOperator() && (CurTok_ != tok_identifier))
    THROW_SYNTAX_ERROR(
        "Expected identifier or operator after ':'.",
        getIdentifierLoc());

  auto OperatorLoc = getIdentifierLoc();
  std::unique_ptr<NodeAST> Expr;

  if (isTokOperator()) {
    Expr = std::make_unique<ReductionStmtAST>(
        getLocationRange(BeginLoc),
        VarIds,
        CurTok_,
        OperatorLoc);
  }
  else {
    auto OperatorStr = getIdentifierStr();
    Expr = std::make_unique<ReductionStmtAST>(
        getLocationRange(BeginLoc),
        VarIds,
        OperatorStr,
        OperatorLoc);
  }
  getNextToken(); // eat identifier

  return Expr;

}

//==============================================================================
// Array expression parser
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseArrayExpr()
{

  auto BeginLoc = getCurLoc();
  getNextToken(); // eat [.

  std::unique_ptr<NodeAST> SizeExpr;
  auto ValExprs = parseExpression();
    
  if (CurTok_ == ';') {
    getNextToken(); // eat ;
    SizeExpr = parseExpression();
  }

  if (CurTok_ != ']')
    THROW_SYNTAX_ERROR(
        "Expected ']'",
        LocationRange(BeginLoc, getCurLoc()) );

 
  // eat ]
  getNextToken();

  return std::make_unique<ArrayExprAST>(
      LocationRange(BeginLoc, getCurLoc()),
      std::move(ValExprs),
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

  //------------------------------------
  // Multi-liner
  if (CurTok_ == '{') {
    getNextToken(); // eat {
    while (CurTok_ != '}') {
      if (CurTok_ == tok_return) {
        getNextToken(); // eat return
        Return = parseExpression();
        break;
      }
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
      if (CurTok_ == tok_sep) getNextToken();
    }
    if (CurTok_ != '}')
      THROW_SYNTAX_ERROR(
          "Only one return statement allowed for a function.",
          getIdentifierLoc() );
    getNextToken(); // eat }
  }
  //------------------------------------
  // One-liner
  else {
    if (CurTok_ == tok_return) {
      getNextToken(); // eat return
      Return = parseExpression();
    }
    else {
      auto E = parseExpression();
      Body.emplace_back( std::move(E) );
    }
  }

  
  if (IsTask) {
    return std::make_unique<TaskAST>(
        std::move(Proto),
        std::move(Body),
        std::move(Return));
  }
  else
    return std::make_unique<FunctionAST>(
        std::move(Proto),
        std::move(Body),
        std::move(Return));
}

//==============================================================================
// prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parsePrototype() {

  std::string FnName;
  auto BeginLoc = getCurLoc();

  auto FnLoc = getIdentifierLoc();

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok_) {
  default:
    THROW_SYNTAX_ERROR(
        "Expected function name in prototype", 
        FnLoc);
  case tok_identifier:
    FnName = getIdentifierStr();
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR(
          "Expected unary operator",
          getIdentifierLoc());
    FnName = "unary";
    FnName += (char)CurTok_;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR(
          "Expected binrary operator",
          getIdentifierLoc());
    FnName = "binary";
    FnName += (char)CurTok_;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok_ == tok_int_literal) {
      auto NumVal = std::atoi(getIdentifierStr().c_str());
      if (NumVal < 1 || NumVal > 100)
        THROW_SYNTAX_ERROR(
            "Invalid precedence of '" << NumVal
            << "' must be between 1 and 100",
            getIdentifierLoc());
      BinaryPrecedence = NumVal;
      getNextToken();
    }
    else {
      THROW_SYNTAX_ERROR(
          "Precedence must be an integer number",
          getIdentifierLoc());
    }
    break;
  }
  
  std::vector<Identifier> ReturnTypes;

  // know it has specified arguments
  if (CurTok_ == ',' || CurTok_ == tok_identifier) {
    ReturnTypes.emplace_back(FnName, FnLoc);
    while (CurTok_ == ',') {
      getNextToken(); // eat ,
      if (CurTok_ != tok_identifier)
        THROW_SYNTAX_ERROR(
            "Expected identifier in return type specification.",
            getLocationRange(BeginLoc));
      ReturnTypes.emplace_back(getIdentifierStr(), getIdentifierLoc());
      getNextToken(); // eat identifier
    }
      
    if (CurTok_ != tok_identifier) {
        THROW_SYNTAX_ERROR(
            "Expected function name specification.",
            getLocationRange(BeginLoc));
    }
    FnName = getIdentifierStr();
    FnLoc = getIdentifierLoc();
    getNextToken();
  }

  
  if (CurTok_ != '(')
    THROW_SYNTAX_ERROR(
        "Expected '(' in prototype",
        getIdentifierLoc());

  getNextToken(); // eat "("

  std::vector<Identifier> Args;
  std::vector<Identifier> ArgTypes;
  std::vector<bool> ArgIsArray;

  while (CurTok_ == tok_identifier) {

    bool IsArray = false;

    auto BeginLoc = getCurLoc();

    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR(
          "Identifier expected n prototype for function '" << FnName << "'",
          getLocationRange(BeginLoc));

    auto TypeLoc = getIdentifierLoc();
    auto TypeName = getIdentifierStr();
    ArgTypes.emplace_back( TypeName, TypeLoc );

    getNextToken(); // eat identifier
    
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR(
          "Mising type or variable name in prototype for function '" 
          << FnName << "'",
          getLocationRange(BeginLoc));
    auto VarName = getIdentifierStr();
    auto VarLoc = getIdentifierLoc();
    
    Args.emplace_back( VarName, VarLoc );
    
    getNextToken(); // eat identifier
    
    if (CurTok_ == '[') {
      IsArray = true;
      auto BeginLoc = getCurLoc();
      getNextToken(); // eat the '['.
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR(
            "Expected ']'",
            getLocationRange(BeginLoc));
      getNextToken(); // eat the ']'
    }
    ArgIsArray.push_back( IsArray );
   
    if (CurTok_ == ',') getNextToken(); // eat ','
  }

  if (CurTok_ != ')')
    THROW_SYNTAX_ERROR(
        "Expected ')' in prototype",
        getIdentifierLoc());

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && Args.size() != Kind)
    THROW_SYNTAX_ERROR(
        "Invalid number of operands for operator: "
        << Kind << " expected, but got " << Args.size(),
        getIdentifierLoc());

  return std::make_unique<PrototypeAST>(
      Identifier{FnName, FnLoc},
      std::move(Args),
      std::move(ArgTypes),
      std::move(ArgIsArray),
      std::move(ReturnTypes),
      Kind != 0,
      BinaryPrecedence);
}

} // namespace
