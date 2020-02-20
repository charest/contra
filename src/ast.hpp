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
/// BaseAST - Base class for all AST nodes
//==============================================================================
class BaseAST {
public:
  // verbosity is on
  static bool IsVerbose;
  
  void echo(const std::string & msg, int Depth) {
    if (IsVerbose) {
      std::cerr << std::string(2*Depth, '+');
      std::cerr << msg << std::endl;
    }
  }
};

//==============================================================================
/// ExprAST - Base class for all expression nodes.
//==============================================================================
class ExprAST : public BaseAST {

  SourceLocation Loc;
  
public:

  ExprAST(SourceLocation Loc) : Loc(Loc) {}

  virtual ~ExprAST() = default;

  virtual Value *codegen(CodeGen &, int) = 0;
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

  Value *codegen(CodeGen &, int Depth=0) override;
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

  Value *codegen(CodeGen &, int Depth=0) override;
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

  Value *codegen(CodeGen &, int Depth=0) override;
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

  Value *codegen(CodeGen &, int Depth=0) override;
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

  Value *codegen(CodeGen &, int Depth=0) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
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

  Value *codegen(CodeGen &, int Depth=0) override;
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

  Value *codegen(CodeGen &, int Depth=0) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// VarExprAST - Expression class for var/in
//==============================================================================
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

public:
  VarExprAST(SourceLocation Loc,
      std::vector<std::pair<std::string,
      std::unique_ptr<ExprAST>>> VarNames)
    : ExprAST(Loc), VarNames(std::move(VarNames)) {}

  Value *codegen(CodeGen &, int Depth=0) override;
  raw_ostream &dump(raw_ostream &out, int ind) override;
};

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
//==============================================================================
class PrototypeAST : public BaseAST {
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

  
  Function *codegen(CodeGen &, int Depth=0);

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

  Function *codegen(CodeGen &, std::map<char, int> &, int Depth=0);
  raw_ostream &dump(raw_ostream &out, int ind);

};

} // namespace

#endif // CONTRA_AST_HPP
