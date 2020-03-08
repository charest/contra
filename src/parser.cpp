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
std::unique_ptr<ExprAST> Parser::parseIntegerExpr() {
  auto NumVal = std::atoi( TheLex_.getIdentifierStr().c_str() );
  auto Result = std::make_unique<IntegerExprAST>(TheLex_.getCurLoc(), NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// numberexpr ::= number
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseRealExpr() {
  auto NumVal = std::atof( TheLex_.getIdentifierStr().c_str() );
  auto Result = std::make_unique<RealExprAST>(TheLex_.getCurLoc(), NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// stringexpr ::= string
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseStringExpr() {
  auto Result = std::make_unique<StringExprAST>(TheLex_.getCurLoc(), TheLex_.getIdentifierStr());
  getNextToken(); // consume the number
  return std::move(Result);
}

//==============================================================================
// parenexpr ::= '(' expression ')'
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseParenExpr() {
  getNextToken(); // eat (.
  auto Loc = getLoc();
  auto V = parseExpression();

  if (CurTok_ != ')') { 
    std::cerr << std::endl;
    TheLex_.barf(std::cerr, Loc);
    std::cerr << std::endl;
    THROW_SYNTAX_ERROR("Expected ')'", Loc.getLine());
  }
  getNextToken(); // eat ).
  return V;
}

//==============================================================================
// identifierexpr
//   ::= 
//   ::= identifier '(' expression* ')'
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseIdentifierExpr() {
  std::string IdName = TheLex_.getIdentifierStr();

  SourceLocation LitLoc = TheLex_.getCurLoc();

  getNextToken(); // eat identifier.

  //----------------------------------------------------------------------------
  // Simple variable ref.
  if (CurTok_ != '(') {
    
    // get variable type
    auto vit = NamedValues_.find(IdName);
    if ( vit == NamedValues_.end() ) {
      std::cerr << std::endl;
      TheLex_.barf(std::cerr, LitLoc);
      std::cerr << std::endl;
      THROW_NAME_ERROR( "Variable '" << IdName << "' was referenced but not defined",
        LitLoc.getLine() );
    }
    
    // variable type
    const auto & TheSymbol = vit->second;
    auto VarType = TheSymbol.getType();

    // array value load
    if (CurTok_ == '[') {
      if (!TheSymbol.isArray())
        THROW_SYNTAX_ERROR( "Previously defined scalar variable '" << IdName
            << "' cannot be referenced as an array" , getLine() );
      getNextToken(); // eat [
      auto Arg = parseExpression();
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR( "Expected ']' at the end of array expression",
            getLine() );
      getNextToken(); // eat ]
      return std::make_unique<VariableExprAST>(LitLoc, IdName, VarType, std::move(Arg));
    }
    // scalar load
    else {
      return std::make_unique<VariableExprAST>(LitLoc, IdName, VarType);
    }

  } // variable reference

  //----------------------------------------------------------------------------
  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok_ != ')') {
    while (true) {
      auto Arg = parseExpression();
      Args.push_back(std::move(Arg));

      if (CurTok_ == ')')
        break;

      if (CurTok_ != ',')
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
std::unique_ptr<ExprAST> Parser::parseIfExpr() {

  ExprLocPairList Conds;
  ExprBlockList BBlocks;
  
  //---------------------------------------------------------------------------
  // If
  {

    auto IfLoc = TheLex_.getCurLoc();
    getNextToken(); // eat the if.

    // condition.
    auto Cond = parseExpression();
    addExpr(Conds, IfLoc, std::move(Cond));

    if (CurTok_ != tok_then) {
      std::cerr << std::endl;
      TheLex_.barf(std::cerr, IfLoc);
      std::cerr << std::endl;
      THROW_SYNTAX_ERROR("Expected 'then' after 'if'", IfLoc.getLine());
    }
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
  
    auto ElifLoc = TheLex_.getCurLoc();
    getNextToken(); // eat elif

    // condition.
    auto Cond = parseExpression();
    addExpr(Conds, ElifLoc, std::move(Cond));
  
    if (CurTok_ != tok_then)
      THROW_SYNTAX_ERROR("Expected 'then' after 'elif'", getLine());
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

    auto ElseLoc = TheLex_.getCurLoc();
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

  return IfExprAST::makeNested( Conds, BBlocks );
}

//==============================================================================
// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseForExpr() {
  auto ForLoc = getLoc();
  getNextToken(); // eat the for.

  auto IdentLoc = getLoc();

  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after 'for'", getLine());
  std::string IdName = TheLex_.getIdentifierStr();
  getNextToken(); // eat identifier.

  auto it = NamedValues_.find(IdName);
  Symbol OldType;
  bool oldsaved = false;

  // override type
  if (it != NamedValues_.end() ) {
    OldType = it->second;
    oldsaved = true;
    it->second = Symbol(VarTypes::Int, IdentLoc);
  }
  // create var
  else {
    NamedValues_.addSymbol( IdName, VarTypes::Int, IdentLoc );
  }
  
  if (CurTok_ != tok_in)
    THROW_SYNTAX_ERROR("Expected 'in' after 'for'", getLine());
  getNextToken(); // eat in

  auto StartLoc = getLoc();
  auto Start = parseExpression();

  ForExprAST::LoopType Loop;
  if (CurTok_ == tok_to) {
    Loop = ForExprAST::LoopType::To;
  }
  else if (CurTok_ == tok_until) {
    Loop = ForExprAST::LoopType::Until;
  }
  else {
    std::cerr << std::endl;
    TheLex_.barf(std::cerr, StartLoc);
    THROW_SYNTAX_ERROR("Expected 'to' after for start value in 'for' loop", StartLoc.getLine());
  }
  getNextToken(); // eat to

  auto EndLoc = getLoc();
  auto End = parseExpression();

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok_ == tok_by) {
    getNextToken();
    Step = parseExpression();
  }

  if (CurTok_ != tok_do) {
    std::cerr << std::endl;
    TheLex_.barf(std::cerr, ForLoc);
    std::cerr << std::string(ForLoc.getCol()-2, ' ') << "do" << std::endl;
    std::cerr << std::endl;
    THROW_SYNTAX_ERROR("Expected 'do' after 'for'", ForLoc.getLine());
  }
  getNextToken(); // eat 'do'.
  
  // add statements
  ExprBlock Body;
  while (CurTok_ != tok_end) {
    auto E = parseExpression();
    Body.emplace_back( std::move(E) );
    if (CurTok_ == tok_sep) getNextToken();
  }
  
  // make a for loop
  auto F = std::make_unique<ForExprAST>(TheLex_.getCurLoc(), IdName, std::move(Start),
      std::move(End), std::move(Step), std::move(Body), Loop);


  it = NamedValues_.find(IdName);
  if (it == NamedValues_.end() )
    THROW_CONTRA_ERROR( "Iterator somehow removed from symbol table" );
  if (oldsaved)
    it->second = OldType;
  else 
    NamedValues_.erase(it);
  
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
std::unique_ptr<ExprAST> Parser::parsePrimary() {
  
  switch (CurTok_) {
  case tok_identifier:
  case tok_int:
  case tok_real:
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
    return parseForExpr();
  case tok_var:
    return parseVarExpr();
  case tok_string:
    return parseStringExpr();
  default:
    THROW_SYNTAX_ERROR("Unknown token '" <<  getTokName(CurTok_)
        << "' when expecting an expression", getLine());
  }
}

//==============================================================================
// binoprhs
//   ::= ('+' primary)*
//==============================================================================
std::unique_ptr<ExprAST>
Parser::parseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS)
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
    SourceLocation BinLoc = TheLex_.getCurLoc();
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
std::unique_ptr<ExprAST> Parser::parseExpression() {
  auto LHS = parseUnary();

  auto RHS = parseBinOpRHS(0, std::move(LHS));
  return std::move(RHS);
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

  SourceLocation FnLoc = TheLex_.getCurLoc();
  auto E = parseExpression();
  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>(FnLoc, "__anon_expr", E->InferredType);
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
std::unique_ptr<ExprAST> Parser::parseUnary() {

  // If the current token is not an operator, it must be a primary expr.
  if (!isascii(CurTok_) || CurTok_ == '(' || CurTok_ == ',') {
    auto P = parsePrimary();
    return std::move(P);
  }

  // If this is a unary operator, read it.
  int Opc = CurTok_;
  getNextToken();
  auto Operand = parseUnary();
  return std::make_unique<UnaryExprAST>(TheLex_.getCurLoc(), Opc, std::move(Operand));
}

//==============================================================================
// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseVarExpr() {

  getNextToken();  // eat the var.
  // At least one variable name is required.
  if (CurTok_ != tok_identifier)
    THROW_SYNTAX_ERROR("Expected identifier after var", getLine());

  std::vector<SourceLocation> VarLoc(1, getLoc());
  std::vector<std::string> VarNames(1, TheLex_.getIdentifierStr());
  getNextToken();  // eat identifier.

  VarTypes VarType;
  bool VarDefined = false;
  bool IsArray = false;

  std::unique_ptr<ExprAST> Size;

  // get additional variables
  while (CurTok_ == ',') {
    getNextToken();  // eat ','  
    if (CurTok_ != tok_identifier)
      THROW_SYNTAX_ERROR("Only variable names are allowed in definition.", getLine());
    VarLoc.push_back(getLoc());
    VarNames.push_back( TheLex_.getIdentifierStr() );
    getNextToken();  // eat identifier
  }

  // read modifiers
  if (CurTok_ == ':') {
    getNextToken(); // eat the ':'.
    
    VarDefined = true;

    if (CurTok_ == '[') {
      IsArray = true;
      getNextToken(); // eat the '['.
    }

    if (CurTok_ == tok_int) {
      VarType = VarTypes::Int;
    }
    else if (CurTok_ == tok_real) {
      VarType = VarTypes::Real;
    }
    else {
      THROW_SYNTAX_ERROR("Variable type '" << TheLex_.getIdentifierStr()
          << "' not supported for '" << VarNames[0] << "'", getLine());
    }
    
    getNextToken(); // eat the 'real'/'int'

    if (IsArray) {


      if (CurTok_ != ']' && CurTok_ != ';')
        THROW_SYNTAX_ERROR("Array definition expected ']' or ';' instead of '"
            << getTokName(CurTok_) << "'", getLine());
      else if (CurTok_ == ';') {
        getNextToken(); // eat ;
        Size = parseExpression();
      }
      
      if (CurTok_ != ']')
        THROW_SYNTAX_ERROR("Array definition must end with ']' instead of '"
            << getTokName(CurTok_) << "'", getLine());

      getNextToken(); // eat [
    }

  }
  
  // Read the optional initializer.
  std::unique_ptr<ExprAST> Init;
  if (CurTok_ == '=') {
    getNextToken(); // eat the '='.
    
    if (CurTok_ == '[')
      Init = parseArrayExpr(VarType);
    else {
      Init = parseExpression();
    }
  }
  else {
    std::cerr << std::endl;
    TheLex_.barf(std::cerr, getLoc());
    std::cerr << std::string(getCol()-2, ' ') << "=" << std::endl;
    std::cerr << std::endl;
    THROW_SYNTAX_ERROR("Variable definition for '" << VarNames << "'"
        << " has no initializer", getLine());
  }

  if (!VarDefined) {
    auto ptr = Init.get();
    if ( dynamic_cast<IntegerExprAST*>(ptr) )
      VarType = VarTypes::Int;
    else if ( dynamic_cast<RealExprAST*>(ptr) )
      VarType = VarTypes::Real;
    else if (ptr->InferredType != VarTypes::Void)
      VarType = ptr->InferredType;
    else
      THROW_SYNTAX_ERROR( "Could not infer variable type for '" << VarNames[0] << "'", getLine() );
  }
  
  for ( int i=0; i<VarNames.size(); ++i )
    NamedValues_.addSymbol( VarNames[i], VarType, VarLoc[i], IsArray );
 
  if (IsArray)
    return std::make_unique<ArrayVarExprAST>(TheLex_.getCurLoc(), VarNames,
        VarType, std::move(Init), std::move(Size));
  else
    return std::make_unique<VarExprAST>(TheLex_.getCurLoc(), VarNames, VarType,
        std::move(Init));
}


//==============================================================================
// Array expression parser
//==============================================================================
std::unique_ptr<ExprAST> Parser::parseArrayExpr(VarTypes VarType)
{

  getNextToken(); // eat [.

  ExprBlock ValExprs;
  std::unique_ptr<ExprAST> SizeExpr;

  while (CurTok_ != ']') {
      auto Loc = getLoc();
    auto E = parseExpression();

    ValExprs.emplace_back( std::move(E) );
    
    if (CurTok_ == ';') {
      getNextToken(); // eat ;
      if (CurTok_ == ']') {
        std::cerr << std::endl;
        TheLex_.barf(std::cerr, Loc);
        std::cerr << std::endl;
        THROW_SYNTAX_ERROR("Expected size expression after ';'", Loc.getLine());
      }
      SizeExpr = std::move(parseExpression());
      break;
    }

    if (CurTok_ == ',') getNextToken();
  }

  if (CurTok_ != ']')
    THROW_SYNTAX_ERROR( "Expected ']'", getLine() );

 
  // eat ]
  getNextToken();

  return std::make_unique<ArrayExprAST>(TheLex_.getCurLoc(), VarType,
      std::move(ValExprs), std::move(SizeExpr));
}


//==============================================================================
// Toplevel function parser
//==============================================================================
std::unique_ptr<FunctionAST> Parser::parseFunction() {

  NamedValues_.clear();

  getNextToken(); // eat def.
  auto Proto = parsePrototype();

  ExprBlock Body;
  std::unique_ptr<ExprAST> Return;

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
        getLine() );
  
  // eat end
  getNextToken();
  
  if (Return)
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body), std::move(Return));
  else
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
}

//==============================================================================
// prototype
//==============================================================================
std::unique_ptr<PrototypeAST> Parser::parsePrototype() {

  std::string FnName;

  SourceLocation FnLoc = TheLex_.getCurLoc();

  unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok_) {
  default:
    THROW_SYNTAX_ERROR("Expected function name in prototype", getLine());
  case tok_identifier:
    FnName = TheLex_.getIdentifierStr();
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR("Expected unary operator", getLine());
    FnName = "unary";
    FnName += (char)CurTok_;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok_))
      THROW_SYNTAX_ERROR("Expected binrary operator", getLine());
    FnName = "binary";
    FnName += (char)CurTok_;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok_ == tok_int_number) {
      auto NumVal = std::atoi(TheLex_.getIdentifierStr().c_str());
      if (NumVal < 1 || NumVal > 100)
        THROW_SYNTAX_ERROR("Invalid precedence of '" << NumVal
            << "' must be between 1 and 100", getLine());
      BinaryPrecedence = NumVal;
      getNextToken();
    }
    else {
      THROW_SYNTAX_ERROR("Precedence must be an integer number", getLine());
    }
    break;
  }
  
  if (CurTok_ != '(')
    THROW_SYNTAX_ERROR("Expected '(' in prototype", getLine());

  getNextToken(); // eat "("

  std::vector< std::pair<std::string, Symbol> > Args;

  while (CurTok_ == tok_identifier) {

    auto Loc = getLoc();
    auto Name = TheLex_.getIdentifierStr();
    bool IsArray = false;
    getNextToken(); // eat identifier
    
    if (CurTok_ != ':') 
      THROW_SYNTAX_ERROR("Variable '" << Name << "' needs a type specifier", getLine());
    getNextToken(); // eat ":"
    
    if (CurTok_ == '[') {
      IsArray = true;
      getNextToken(); // eat the '['.
    }
    
    auto VarType = getVarType(CurTok_);
    if (VarType == VarTypes::Void)
      THROW_SYNTAX_ERROR("Variable '" << Name << "' of type '"
          << getVarTypeName(VarType) << "' is not allowed "
          << "in a function prototype", getLine());
    getNextToken(); // eat vartype
    
    if (IsArray) {
      if (CurTok_ != ']') 
        THROW_SYNTAX_ERROR("Array declaration expected ']' instead of '"
          << getTokName(CurTok_) << "'", getLine());
      getNextToken(); // eat ]
    }

    Args.emplace_back( Name, Symbol(VarType, Loc, IsArray) );

    if (CurTok_ == ',') getNextToken(); // eat ','
  }

  if (CurTok_ != ')')
    THROW_SYNTAX_ERROR("Expected ')' in prototype", getLine());

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && Args.size() != Kind)
    THROW_SYNTAX_ERROR("Invalid number of operands for operator: "
        << Kind << " expected, but got " << Args.size(), getLine());

  // add these varaibles to the current parser scope
  NamedValues_.addSymbols( Args );

  VarTypes ReturnType = VarTypes::Void;
  if (CurTok_ == '-') {
    getNextToken(); // eat -
    if (CurTok_ != '>')
      THROW_SYNTAX_ERROR("Expected '>' after '-' for return statements", getLine());
    getNextToken(); // eat >

    auto VarType = getVarType(CurTok_);
    if (VarType == VarTypes::Void)
      THROW_SYNTAX_ERROR("Return type '"
          << getVarTypeName(VarType) << "' is not allowed "
          << "in a function prototype", getLine());
    getNextToken(); // eat vartype

    ReturnType = VarType;
  }


  return std::make_unique<PrototypeAST>(FnLoc, FnName,
      std::move(Args), ReturnType, Kind != 0, BinaryPrecedence);
}

} // namespace
