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
    if (CurTok_ != ')') Args = std::move(parseExpression());
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
      auto IndexExpr = std::move(parseExpression());
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
#if 0
    //----------------------------------
    // field
    else if (CurTok_ == '{') {
      auto FieldLoc = getCurLoc();
      getNextToken(); // eat {
      if (CurTok_ != tok_identifier)
        THROW_SYNTAX_ERROR(
            "Expected identifier in field declaration.",
            getIdentifierLoc());
      auto IndexExpr = std::move(parseExpression());
      getNextToken(); // eat identifier
      if (CurTok_ != '}')
        THROW_SYNTAX_ERROR(
            "Expected '}'  in field declaration.",
            getLocationRange(FieldLoc));
      getNextToken(); // eat }
      return std::make_unique<FieldDeclExprAST>(
          getLocationRange(BeginLoc),
          Id,
          std::move(IndexExpr),
          VarTypeId);
    }
#endif
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
  
    auto ElifLoc = getIdentifierLoc();
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
  auto ForLoc = getIdentifierLoc();
  
  bool IsForEach = CurTok_ == tok_foreach;

  getNextToken(); // eat the for.

  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after 'for'", getIdentifierLoc());
  std::string IdName = getIdentifierStr();
  auto IdentLoc = getIdentifierLoc();
  getNextToken(); // eat identifier.

  if (CurTok_ != tok_in)
    THROW_SYNTAX_ERROR("Expected 'in' after 'for'", getIdentifierLoc());
  getNextToken(); // eat in

  auto StartLoc = getCurLoc();
  auto Start = parseExpression();

  ForStmtAST::LoopType Loop;
  if (CurTok_ == tok_to) {
    Loop = ForStmtAST::LoopType::To;
  }
  else if (CurTok_ == tok_until) {
    Loop = ForStmtAST::LoopType::Until;
  }
  else if (CurTok_ == tok_do ) {
    Loop = ForStmtAST::LoopType::Range;
  }
  else
    THROW_SYNTAX_ERROR(
        "Expected 'to' after for start value in 'for' loop", 
        LocationRange(StartLoc, getCurLoc()));

  getNextToken(); // eat to/do

  std::unique_ptr<NodeAST> End, Step;
  if ( Loop != ForStmtAST::LoopType::Range ) {
    End = parseExpression();

    // The step value is optional.
    if (CurTok_ == tok_by) {
      getNextToken();
      Step = parseExpression();
    }

    if (CurTok_ != tok_do)
      THROW_SYNTAX_ERROR("Expected 'do' after 'for'", getIdentifierLoc());
    getNextToken(); // eat 'do'.
  }
  
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
    F = std::make_unique<ForeachStmtAST>(ForLoc, Id, std::move(Start),
      std::move(End), std::move(Step), std::move(Body), Loop);
  else
    F = std::make_unique<ForStmtAST>(ForLoc, Id, std::move(Start),
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
  case tok_real_literal:
    return parseRealExpr();
  case tok_int_literal:
    return parseIntegerExpr();
  case '(':
    return parseParenExpr();
  case '[':
    return parseArrayExpr();
  case '{':
    return parseRangeExpr();
  case tok_if:
    return parseIfExpr();
  case tok_for:
  case tok_foreach:
    return parseForExpr();
  case tok_part:
  case tok_use:
    return parsePartitionExpr();
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

#if 0
//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseVarDefExpr() {


  auto BeginVarLoc = getCurLoc();

  // At least one variable name is required.
  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after type", getIdentifierLoc());

  while (true) {
    
    VarDeclAST::VariableInfo Var;

    Var.Id = Identifier(getIdentifierStr(), getIdentifierLoc());
    getNextToken();  // eat identifier.

    //----------------------------------
    // Array
    if (CurTok_ == '[') {
      VarAttr = VarDeclAST::AttrType::Array;
      auto BeginLoc = getCurLoc();
      getNextToken(); // eat [
      if (CurTok_ != ']') Var.SizeExpr = std::move(parseExpression());
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR(
            "Expected ']'  in array declaration.",
            LocationRange(BeginLoc, getCurLoc()));
      getNextToken(); // eat ]
    }
    //----------------------------------
    // field
    else if (CurTok_ == '{') {
      VarAttr = VarDeclAST::AttrType::Field;
      auto BeginLoc = getCurLoc();
      getNextToken(); // eat {
      if (CurTok_ != tok_identifier)
        THROW_SYNTAX_ERROR(
            "Expected identifier in field declaration.",
            getIdentifierLoc());
      Var.IndexExpr = std::move(parseExpression());
      getNextToken(); // eat identifier
      if (CurTok_ != '}')
        THROW_SYNTAX_ERROR(
            "Expected '}'  in field declaration.",
            LocationRange(BeginLoc, getCurLoc()));
      getNextToken(); // eat }
    }

    Vars.emplace_back(Var);

   if (CurTok_ != ',') break;
  }
  
  auto EndVarLoc = getCurLoc();
  
  // Read the optional initializer.
  std::unique_ptr<NodeAST> Init;
  if (CurTok_ == tok_asgmt) {
    getNextToken(); // eat the '='.
    Init = parseExpression();
  }
  else {
    std::vector<std::string> Names;
    for ( const auto & V : Vars ) Names.emplace_back(V.Id.getName());
    THROW_SYNTAX_ERROR("Variable definition for '" << Names << "'"
        << " has no initializer",
        LocationRange(BeginVarLoc, EndVarLoc) );
  }

  return std::make_unique<VarDeclAST>(
      TypeLoc,
      Vars,
      VarType,
      std::move(Init));
}
#endif

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<NodeAST> Parser::parsePartitionExpr() {

  bool IsUse = CurTok_ == tok_use;

  auto Loc = getIdentifierLoc();
  getNextToken();  // eat the partition

  if (IsUse) {

    auto ColorLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'use'.", ColorLoc);
    auto ColorName = getIdentifierStr();
    getNextToken(); // eat identifier.

    auto ForLoc = getIdentifierLoc();
    if (CurTok_ != tok_for)
      THROW_SYNTAX_ERROR("Expected 'for' after identifier.", ForLoc);
    getNextToken(); // eat for.

    auto RangeLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'for'.", RangeLoc);
    auto RangeName = getIdentifierStr();
    getNextToken(); // eat identifier.

    return std::make_unique<PartitionStmtAST>(
        Loc,
        Identifier{RangeName, RangeLoc},
        Identifier{ColorName, ColorLoc});
  } 
  else {

    auto RangeLoc = getIdentifierLoc();
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Expected an identifier after keyword 'partition'.", RangeLoc);
    auto RangeName = getIdentifierStr();
    getNextToken(); // eat identifier.

    auto ByLoc = getCurLoc();
    if (CurTok_ != tok_by)
      THROW_SYNTAX_ERROR("Expected 'by' after identifier.", ByLoc);
    getNextToken(); // eat by.

    auto ColorExpr = parseExpression();

    // if where is included
    ASTBlock Body;
    if (CurTok_ == tok_where) {
      getNextToken(); // eat where
      while (CurTok_ != tok_end) {
        auto E = parseExpression();
        Body.emplace_back( std::move(E) );
        if (CurTok_ == tok_sep) getNextToken();
      }
      getNextToken(); // eat end
    }

    return std::make_unique<PartitionStmtAST>(
        Loc,
        Identifier{RangeName, RangeLoc},
        std::move(ColorExpr),
        std::move(Body));
  }
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
    SizeExpr = std::move(parseExpression());
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
// Array expression parser
//==============================================================================
std::unique_ptr<NodeAST> Parser::parseRangeExpr()
{

  auto BeginLoc = getCurLoc();
  getNextToken(); // eat {.

  auto StartExpr = parseExpression();

  if (CurTok_ != tok_range && CurTok_ != ',')
      THROW_SYNTAX_ERROR(
          "Expected '..' or ',' in range expression.",
          LocationRange(BeginLoc, getCurLoc()));
  getNextToken(); // eat ..
  
  auto EndExpr = parseExpression();
  
  if (CurTok_ != '}')
    THROW_SYNTAX_ERROR(
        "Expected '}' at the end of a range expression.",
        LocationRange(BeginLoc, getCurLoc()));
  getNextToken(); // eat }

  return std::make_unique<RangeExprAST>(
      LocationRange(BeginLoc, getCurLoc()),
        std::move(StartExpr),
        std::move(EndExpr));
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
