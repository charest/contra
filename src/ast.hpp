#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "codegen.hpp"
#include "sourceloc.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace contra {

class Parser;
using llvm::raw_ostream;

//==============================================================================
/// ExprAST - Base class for all expression nodes.
//==============================================================================
class ExprAST {

  SourceLocation Loc;

public:
  ExprAST(SourceLocation Loc) : Loc(Loc) {}
  virtual ~ExprAST() = default;
  virtual Value *codegen(CodeGen &) = 0;
  int getLine() const { return Loc.Line; }
  int getCol() const { return Loc.Col; }
  virtual raw_ostream &dump(raw_ostream &out, int ind) {
    return out << ':' << getLine() << ':' << getCol() << '\n';
  }
};

//==============================================================================
/// NumberExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(SourceLocation Loc, double Val) : ExprAST(Loc), Val(Val) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
  
};

//==============================================================================
/// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(SourceLocation Loc, const std::string &Name) :
    ExprAST(Loc), Name(Name) {}

  Value *codegen(CodeGen &) override;
  const std::string &getName() const { return Name; }
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
      char Op, std::unique_ptr<ExprAST> LHS,
      std::unique_ptr<ExprAST> RHS)
    : ExprAST(Loc), Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS))
  {}

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
  std::unique_ptr<ExprAST> Cond, Then, Else;

public:
  IfExprAST(SourceLocation Loc,
      std::unique_ptr<ExprAST> Cond,
      std::unique_ptr<ExprAST> Then,
      std::unique_ptr<ExprAST> Else)
    : ExprAST(Loc), Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else))
  {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
// ForExprAST - Expression class for for/in.
//==============================================================================
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
  ForExprAST(SourceLocation Loc,
      const std::string &VarName,
      std::unique_ptr<ExprAST> Start,
      std::unique_ptr<ExprAST> End,
      std::unique_ptr<ExprAST> Step,
      std::unique_ptr<ExprAST> Body)
    : ExprAST(Loc), VarName(VarName), Start(std::move(Start)), End(std::move(End)),
    Step(std::move(Step)), Body(std::move(Body))
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
    : ExprAST(Loc), Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// VarExprAST - Expression class for var/in
//==============================================================================
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::unique_ptr<ExprAST> Body;

public:
  VarExprAST(SourceLocation Loc,
      std::vector<std::pair<std::string,
      std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body)
    : ExprAST(Loc), VarNames(std::move(VarNames)), Body(std::move(Body)) {}

  Value *codegen(CodeGen &) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
//==============================================================================
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;
  bool IsOperator;
  unsigned Precedence;  // Precedence if a binary op.
  int Line;

public:
  PrototypeAST(
    SourceLocation Loc,
    const std::string &Name,
    std::vector<std::string> Args,
    bool IsOperator = false,
    unsigned Prec = 0)
      : Name(Name), Args(std::move(Args)), IsOperator(IsOperator),
        Precedence(Prec), Line(Loc.Line)
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
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}

  Function *codegen(CodeGen &, Parser &);
  raw_ostream &dump(raw_ostream &out, int ind);

};

} // namespace

#endif // CONTRA_AST_HPP
