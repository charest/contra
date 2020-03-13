#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "codegen.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "expression.hpp"
#include "identifier.hpp"
#include "sourceloc.hpp"

#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace contra {

class Parser;
class AstDispatcher;

//==============================================================================
/// NodeAST - Base class for all nodes.
//==============================================================================
class NodeAST {
public:
  virtual ~NodeAST() = default;
  virtual void accept(AstDispatcher& dispatcher) = 0;

};


//==============================================================================
/// ExprAST - Base class for all expression nodes.
//==============================================================================
class ExprAST : public NodeAST {

  SourceLocation Loc_;
  
public:
  
  ExprAST(SourceLocation Loc) : Loc_(Loc) {}

  virtual ~ExprAST() = default;
  
  auto getLoc() const { return Loc_; }
  int getLine() const { return Loc_.getLine(); }
  int getCol() const { return Loc_.getCol(); }

};

//==============================================================================
/// ValueExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
template< typename T >
class ValueExprAST : public ExprAST {
protected:

  T Val_;

public:
  ValueExprAST(SourceLocation Loc, T Val)
    : ExprAST(Loc), Val_(Val) {}

  const T & getVal() const { return Val_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
  
};

// alias for ints
using IntegerExprAST = ValueExprAST<int_t>;
// alias for reals
using RealExprAST = ValueExprAST<real_t>;
// alias for strings
using StringExprAST = ValueExprAST<std::string>;

//==============================================================================
/// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
class VariableExprAST : public ExprAST {
protected:

  std::string Name_;
  std::unique_ptr<ExprAST> IndexExpr_;

public:

  VariableExprAST(SourceLocation Loc, const std::string &Name)
    : ExprAST(Loc), Name_(Name)
  {}

  VariableExprAST(SourceLocation Loc, const std::string &Name,
      std::unique_ptr<ExprAST> IndexExpr)
    : ExprAST(Loc), Name_(Name), IndexExpr_(std::move(IndexExpr))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  const std::string &getName() const { return Name_; }
  
  bool isArray() const { return static_cast<bool>(IndexExpr_); }
  
  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
/// ArrayExprAST - Expression class for referencing an array.
//==============================================================================
class ArrayExprAST : public ExprAST {
protected:

  ExprBlock ValExprs_;
  std::unique_ptr<ExprAST> SizeExpr_;

public:

  ArrayExprAST(SourceLocation Loc, ExprBlock Vals,
      std::unique_ptr<ExprAST> Size)
    : ExprAST(Loc), ValExprs_(std::move(Vals)), SizeExpr_(std::move(Size))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;

};

//==============================================================================
/// CastExprAST - Expression class for casts
//==============================================================================
class CastExprAST : public ExprAST {
protected:

  std::unique_ptr<ExprAST> FromExpr_;
  Identifier TypeId_;


public:
  CastExprAST(SourceLocation Loc, std::unique_ptr<ExprAST> FromExpr,
      Identifier TypeId) : ExprAST(Loc), FromExpr_(std::move(FromExpr)),
      TypeId_(TypeId)
  {}

  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
  
};


//==============================================================================
/// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
class UnaryExprAST : public ExprAST {
protected:

  char OpCode_;
  std::unique_ptr<ExprAST> OpExpr_;

public:
  UnaryExprAST(SourceLocation Loc,
      char Opcode,
      std::unique_ptr<ExprAST> Operand)
    : ExprAST(Loc), OpCode_(Opcode), OpExpr_(std::move(Operand))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
/// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
class BinaryExprAST : public ExprAST {
protected:

  char OpCode_;
  std::unique_ptr<ExprAST> LeftExpr_;
  std::unique_ptr<ExprAST> RightExpr_;

public:
  BinaryExprAST(SourceLocation Loc, 
      char Op, std::unique_ptr<ExprAST> lhs,
      std::unique_ptr<ExprAST> rhs)
    : ExprAST(Loc), OpCode_(Op), LeftExpr_(std::move(lhs)), RightExpr_(std::move(rhs))
  {}

  char getOperand() const { return OpCode_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
/// CallExprAST - Expression class for function calls.
//==============================================================================
class CallExprAST : public ExprAST {
protected:
  
  std::string Callee_;
  std::vector<std::unique_ptr<ExprAST>> ArgExprs_;

public:

  CallExprAST(SourceLocation Loc,
      const std::string &Callee,
      std::vector<std::unique_ptr<ExprAST>> Args)
    : ExprAST(Loc), Callee_(Callee), ArgExprs_(std::move(Args))
  {}

  const std::string & getCalleeName() const { return Callee_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
  
};

//==============================================================================
/// IfExprAST - Expression class for if/then/else.
//==============================================================================
class IfExprAST : public ExprAST {
protected:

  std::unique_ptr<ExprAST> CondExpr_;
  ExprBlock ThenExpr_, ElseExpr_;

public:

  IfExprAST(SourceLocation Loc, std::unique_ptr<ExprAST> Cond,
       ExprBlock Then)
    : ExprAST(Loc), CondExpr_(std::move(Cond)), ThenExpr_(std::move(Then))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  static std::unique_ptr<ExprAST> makeNested( 
    ExprLocPairList & Conds, ExprBlockList & Blocks );
  
  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
// ForExprAST - Expression class for for/in.
//==============================================================================
class ForExprAST : public ExprAST {

public:

  enum class LoopType {
    To, Until
  };

protected:

  Identifier VarId_;
  std::unique_ptr<ExprAST> StartExpr_, EndExpr_, StepExpr_;
  ExprBlock BodyExprs_;
  LoopType Loop_;

public:

  ForExprAST(SourceLocation Loc,
      const Identifier &VarId,
      std::unique_ptr<ExprAST> Start,
      std::unique_ptr<ExprAST> End,
      std::unique_ptr<ExprAST> Step,
      ExprBlock Body,
      LoopType Loop = LoopType::To)
    : ExprAST(Loc), VarId_(VarId), StartExpr_(std::move(Start)),
      EndExpr_(std::move(End)), StepExpr_(std::move(Step)), BodyExprs_(std::move(Body)),
      Loop_(Loop)
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
/// VarDefExprAST - Expression class for var/in
//==============================================================================
class VarDefExprAST : public ExprAST {

protected:

  std::vector<Identifier> VarIds_;
  Identifier TypeId_;
  std::unique_ptr<ExprAST> InitExpr_;

public:

  VarDefExprAST(SourceLocation Loc, const std::vector<Identifier> & Vars, 
      Identifier VarType, std::unique_ptr<ExprAST> Init)
    : ExprAST(Loc), VarIds_(Vars), TypeId_(VarType),
      InitExpr_(std::move(Init)) 
  {}

  virtual bool isArray() const { return false; }
  
  virtual void accept(AstDispatcher& dispatcher) override;
 
  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
/// ArrayDefExprAST - Expression class for var/in
//==============================================================================
class ArrayDefExprAST : public VarDefExprAST {
protected:

  std::unique_ptr<ExprAST> SizeExpr_;

public:

  ArrayDefExprAST(SourceLocation Loc, const std::vector<Identifier> & VarNames, 
      Identifier VarType, std::unique_ptr<ExprAST> Init,
      std::unique_ptr<ExprAST> Size)
    : VarDefExprAST(Loc, VarNames, VarType, std::move(Init)),
      SizeExpr_(std::move(Size))
  {}
  
  virtual bool isArray() const { return true; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
  
};

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
//==============================================================================
class PrototypeAST : public NodeAST {
protected:

  std::string Name_;
  std::unique_ptr<Identifier> ReturnTypeId_;
  bool IsOperator_ = false;
  unsigned Precedence_ = 0;  // Precedence if a binary op.
  SourceLocation Loc_;
  
  std::vector<Identifier> ArgIds_;
  std::vector<Identifier> ArgTypeIds_;
  std::vector<bool> ArgIsArray_;

public:
  
  PrototypeAST(
    SourceLocation Loc,
    const std::string &Name)
      : Name_(Name), Loc_(Loc)
  {}

  PrototypeAST(
    SourceLocation Loc,
    const std::string &Name,
    std::vector<Identifier> && Args,
    std::vector<Identifier> && ArgTypes,
    std::vector<bool> && ArgIsArray,
    std::unique_ptr<Identifier> Return,
    bool IsOperator = false,
    unsigned Prec = 0)
      : Name_(Name), ReturnTypeId_(std::move(Return)), IsOperator_(IsOperator),
        Precedence_(Prec), Loc_(Loc), ArgIds_(std::move(Args)),
        ArgTypeIds_(std::move(ArgTypes)), ArgIsArray_(std::move(ArgIsArray))
  {}

  
  virtual void accept(AstDispatcher& dispatcher) override;
  
  const std::string &getName() const { return Name_; }

  bool isUnaryOp() const { return IsOperator_ && ArgIds_.size() == 1; }
  bool isBinaryOp() const { return IsOperator_ && ArgIds_.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name_[Name_.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence_; }
  auto getLoc() const { return Loc_; }
  
  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
class FunctionAST : public NodeAST {
protected:

  std::unique_ptr<PrototypeAST> ProtoExpr_;
  ExprBlock BodyExprs_;
  std::unique_ptr<ExprAST> ReturnExpr_;

public:

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ExprBlock Body)
      : ProtoExpr_(std::move(Proto)), BodyExprs_(std::move(Body)) {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ExprBlock Body, 
      std::unique_ptr<ExprAST> Return)
      : ProtoExpr_(std::move(Proto)), BodyExprs_(std::move(Body)),
        ReturnExpr_(std::move(Return))
  {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Return)
      : ProtoExpr_(std::move(Proto)), ReturnExpr_(std::move(Return))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;

};

} // namespace

#endif // CONTRA_AST_HPP
