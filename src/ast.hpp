#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "codegen.hpp"
#include "errors.hpp"
#include "sourceloc.hpp"
#include "symbols.hpp"
#include "vartype.hpp"

#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace contra {

class Parser;
using llvm::raw_ostream;

//==============================================================================
/// BaseAST - Base class for all AST nodes
//==============================================================================
class BaseAST {

protected:
  using Value = llvm::Value;
  using Function = llvm::Function;
  using FunctionCallee = llvm::FunctionCallee;

};

//==============================================================================
/// ExprAST - Base class for all expression nodes.
//==============================================================================
class ExprAST : public BaseAST {

  SourceLocation Loc;
  
public:
  
  VarTypes InferredType;

  ExprAST(SourceLocation Loc, VarTypes Type = VarTypes::Void)
    : Loc(Loc), InferredType(Type) {}

  virtual ~ExprAST() = default;

  virtual Value *codegen(CodeGen &) = 0;
  auto getLoc() const { return Loc; }
  int getLine() const { return Loc.getLine(); }
  int getCol() const { return Loc.getCol(); }
  virtual raw_ostream &dump(raw_ostream &out, int ind) {
    return out << ':' << getLine() << ':' << getCol() << '\n';
  }
};

//==============================================================================
/// IntegerExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
class IntegerExprAST : public ExprAST {
  int Val;

public:
  IntegerExprAST(SourceLocation Loc, int Val)
    : ExprAST(Loc, VarTypes::Int), Val(Val) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
  
};

//==============================================================================
/// RealExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
class RealExprAST : public ExprAST {
  double Val;

public:
  RealExprAST(SourceLocation Loc, double Val)
    : ExprAST(Loc, VarTypes::Real), Val(Val) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
  
};

//==============================================================================
/// StringExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
class StringExprAST : public ExprAST {
  std::string Val;

public:
  StringExprAST(SourceLocation Loc, std::string Val)
    : ExprAST(Loc, VarTypes::String), Val(Val) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
  
};

//==============================================================================
/// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  std::unique_ptr<ExprAST> Index;

  VariableExprAST(SourceLocation Loc, const std::string &Name,
      VarTypes Type) : ExprAST(Loc, Type), Name(Name) {}

  Value *codegen(CodeGen &) override;
  const std::string &getName() const { return Name; }
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// ArrayExprAST - Expression class for referencing an array.
//==============================================================================
class ArrayExprAST : public ExprAST {

public:
  std::vector< std::unique_ptr<ExprAST> > Body;
  std::unique_ptr<ExprAST> Repeat;

  Value* SizeExpr = nullptr;

  ArrayExprAST(SourceLocation Loc, VarTypes VarType)
    : ExprAST(Loc, VarType) {}

  Value *codegen(CodeGen &) override
  { THROW_CONTRA_ERROR("Should not be called"); };

  std::pair<llvm::AllocaInst*, Value*> special_codegen(const std::string &,
      CodeGen &);

  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(SourceLocation Loc, 
      char Op, std::unique_ptr<ExprAST> lhs,
      std::unique_ptr<ExprAST> rhs)
    : ExprAST(Loc), Op(Op), LHS(std::move(lhs)), RHS(std::move(rhs))
  {
    // promote lesser types
    if (LHS->InferredType == VarTypes::Real || RHS->InferredType == VarTypes::Real)
      InferredType = VarTypes::Real;
    else if (LHS->InferredType != RHS->InferredType)
      THROW_SYNTAX_ERROR( "Don't know how to handle binary expression with '"
          << getVarTypeName(LHS->InferredType) << "' and '"
          << getVarTypeName(RHS->InferredType) << "'", Loc.getLine() );
    else
      InferredType =  LHS->InferredType;
  }

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
  
};

//==============================================================================
/// CallExprAST - Expression class for function calls.
//==============================================================================
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(SourceLocation Loc,
      const std::string &Callee,
      std::vector<std::unique_ptr<ExprAST>> Args)
    : ExprAST(Loc), Callee(Callee), Args(std::move(Args))
  {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
  
};

//==============================================================================
/// IfExprAST - Expression class for if/then/else.
//==============================================================================
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond;

public:
  std::vector<std::unique_ptr<ExprAST>> Then, Else;

  IfExprAST(SourceLocation Loc, std::unique_ptr<ExprAST> Cond)
    : ExprAST(Loc), Cond(std::move(Cond))
  {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;

  static std::unique_ptr<ExprAST> make( 
    std::list< std::pair< SourceLocation, std::unique_ptr<ExprAST> > > & Conds,
    std::list< std::vector< std::unique_ptr<ExprAST> > > & Blocks );
};

//==============================================================================
// ForExprAST - Expression class for for/in.
//==============================================================================
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step;

public:
  std::vector<std::unique_ptr<ExprAST>> Body;

  ForExprAST(SourceLocation Loc,
      const std::string &VarName,
      std::unique_ptr<ExprAST> Start,
      std::unique_ptr<ExprAST> End,
      std::unique_ptr<ExprAST> Step)
    : ExprAST(Loc), VarName(VarName), Start(std::move(Start)), End(std::move(End)),
    Step(std::move(Step))
  {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(SourceLocation Loc,
      char Opcode,
      std::unique_ptr<ExprAST> Operand)
    : ExprAST(Loc, Operand->InferredType), Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// VarExprAST - Expression class for var/in
//==============================================================================
class VarExprAST : public ExprAST {
  std::vector<std::string> VarNames;
  VarTypes VarType;
  bool IsArray = false;
  std::unique_ptr<ExprAST> Init;

public:
  std::unique_ptr<ExprAST> Size;

  VarExprAST(SourceLocation Loc, const std::vector<std::string> & VarNames, 
      VarTypes VarType, bool IsArray, std::unique_ptr<ExprAST> Init)
    : ExprAST(Loc, Init->InferredType), VarNames(VarNames), VarType(VarType),
      IsArray(IsArray), Init(std::move(Init)) 
  {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
//==============================================================================
class PrototypeAST : public BaseAST {
  std::string Name;
  VarTypes Return;
  bool IsOperator;
  unsigned Precedence;  // Precedence if a binary op.
  int Line;

public:
  
  std::vector< std::pair<std::string, Symbol> > Args;

  PrototypeAST(
    SourceLocation Loc,
    const std::string &Name,
    std::vector< std::pair<std::string, Symbol> > && Args,
    VarTypes Return,
    bool IsOperator = false,
    unsigned Prec = 0)
      : Name(Name), Return(Return), IsOperator(IsOperator),
        Precedence(Prec), Line(Loc.getLine()), Args(std::move(Args))
  {}

  
  Function *codegen(CodeGen &);

  const std::string &getName() const { return Name; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence; }
  int getLine() const { return Line; }
};

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
class FunctionAST : public BaseAST {
  std::unique_ptr<PrototypeAST> Proto;

public:
  std::vector<std::unique_ptr<ExprAST>> Body;
  std::unique_ptr<ExprAST> Return;

  FunctionAST(std::unique_ptr<PrototypeAST> Proto)
      : Proto(std::move(Proto)) {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Return(std::move(Body))
  {}

  Function *codegen(CodeGen &, std::map<char, int> &);
  raw_ostream &dump(raw_ostream &out, int ind);

};

} // namespace

#endif // CONTRA_AST_HPP
