#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "dispatcher.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "identifier.hpp"
#include "sourceloc.hpp"
#include "symbols.hpp"

#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace contra {

class Parser;
class AstDispatcher;

////////////////////////////////////////////////////////////////////////////////
/// NodeAST - Base class for all nodes.
////////////////////////////////////////////////////////////////////////////////
class NodeAST {
  
  SourceLocation Loc_;

public:
  
  NodeAST(const SourceLocation & Loc) : Loc_(Loc) {}
  
  virtual ~NodeAST() = default;

  virtual void accept(AstDispatcher& dispatcher) = 0;
  
  const auto & getLoc() const { return Loc_; }
  int getLine() const { return Loc_.getLine(); }
  int getCol() const { return Loc_.getCol(); }

};

// some useful types
using ASTBlock = std::vector< std::unique_ptr<NodeAST> >;
using ASTBlockList = std::list<ASTBlock>;

inline auto createBlock( ASTBlockList & list)
{ return list.emplace( list.end(), ASTBlock{} ); }


////////////////////////////////////////////////////////////////////////////////
/// ExprAST - Base class for all expression nodes.
////////////////////////////////////////////////////////////////////////////////
class ExprAST : public NodeAST {

  VariableType Type_;
  
public:
  
  ExprAST(const SourceLocation & Loc) : NodeAST(Loc) {}

  virtual ~ExprAST() = default;
  
  void setType(const VariableType & Type) { Type_ = Type; }
  const VariableType getType() const { return Type_; }

};

//==============================================================================
/// ValueExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
template< typename T >
class ValueExprAST : public ExprAST {
protected:

  T Val_;

public:
  ValueExprAST(const SourceLocation & Loc, T Val)
    : ExprAST(Loc), Val_(Val) {}

  const T & getVal() const { return Val_; }
  
  void accept(AstDispatcher& dispatcher) override
  { dispatcher.dispatch(*this); }

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
  std::unique_ptr<NodeAST> IndexExpr_;
  VariableType Type_;

public:

  VariableExprAST(const SourceLocation & Loc, const std::string &Name)
    : ExprAST(Loc), Name_(Name)
  {}

  VariableExprAST(const SourceLocation & Loc, const std::string &Name,
      std::unique_ptr<NodeAST> IndexExpr)
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

  ASTBlock ValExprs_;
  std::unique_ptr<NodeAST> SizeExpr_;

public:

  ArrayExprAST(const SourceLocation & Loc, ASTBlock Vals,
      std::unique_ptr<NodeAST> Size)
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

  std::unique_ptr<NodeAST> FromExpr_;
  Identifier TypeId_;


public:
  CastExprAST(const SourceLocation & Loc, std::unique_ptr<NodeAST> FromExpr,
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
  std::unique_ptr<NodeAST> OpExpr_;

public:
  UnaryExprAST(const SourceLocation & Loc,
      char Opcode,
      std::unique_ptr<NodeAST> Operand)
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
  std::unique_ptr<NodeAST> LeftExpr_;
  std::unique_ptr<NodeAST> RightExpr_;

public:
  BinaryExprAST(const SourceLocation & Loc, 
      char Op, std::unique_ptr<NodeAST> lhs,
      std::unique_ptr<NodeAST> rhs)
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
  ASTBlock ArgExprs_;

public:

  CallExprAST(const SourceLocation & Loc,
      const std::string &Callee,
      ASTBlock Args)
    : ExprAST(Loc), Callee_(Callee), ArgExprs_(std::move(Args))
  {}

  const std::string & getCalleeName() const { return Callee_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
  
};

////////////////////////////////////////////////////////////////////////////////
/// StmtAST - Base class for all statement nodes.
////////////////////////////////////////////////////////////////////////////////
class StmtAST : public NodeAST {
  
public:
  
  StmtAST(const SourceLocation & Loc) : NodeAST(Loc) {}

  virtual ~StmtAST() = default;

};


//==============================================================================
/// IfExprAST - Expression class for if/then/else.
//==============================================================================
class IfStmtAST : public StmtAST {
protected:

  std::unique_ptr<NodeAST> CondExpr_;
  ASTBlock ThenExpr_, ElseExpr_;

public:

  IfStmtAST(const SourceLocation & Loc, std::unique_ptr<NodeAST> Cond,
       ASTBlock Then)
    : StmtAST(Loc), CondExpr_(std::move(Cond)), ThenExpr_(std::move(Then))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  static std::unique_ptr<NodeAST> makeNested( 
    std::list< std::pair<SourceLocation, std::unique_ptr<NodeAST>> > & Conds,
    ASTBlockList & Blocks );
  
  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

//==============================================================================
// ForExprAST - Expression class for for/in.
//==============================================================================
class ForStmtAST : public StmtAST {

public:

  enum class LoopType {
    To, Until
  };

protected:

  Identifier VarId_;
  std::unique_ptr<NodeAST> StartExpr_, EndExpr_, StepExpr_;
  ASTBlock BodyExprs_;
  LoopType Loop_;

public:

  ForStmtAST(const SourceLocation & Loc,
      const Identifier &VarId,
      std::unique_ptr<NodeAST> Start,
      std::unique_ptr<NodeAST> End,
      std::unique_ptr<NodeAST> Step,
      ASTBlock Body,
      LoopType Loop = LoopType::To)
    : StmtAST(Loc), VarId_(VarId), StartExpr_(std::move(Start)),
      EndExpr_(std::move(End)), StepExpr_(std::move(Step)), BodyExprs_(std::move(Body)),
      Loop_(Loop)
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

////////////////////////////////////////////////////////////////////////////////
/// ExprAST - Base class for all expression nodes.
////////////////////////////////////////////////////////////////////////////////
class DeclAST : public NodeAST {

  VariableType Type_;
  
public:
  
  DeclAST(const SourceLocation & Loc) : NodeAST(Loc) {}

  virtual ~DeclAST() = default;
  
  void setType(const VariableType & Type) { Type_ = Type; }
  const VariableType getType() const { return Type_; }

};


//==============================================================================
/// VarDefExprAST - Expression class for var/in
//==============================================================================
class VarDeclAST : public DeclAST {

protected:

  std::vector<Identifier> VarIds_;
  Identifier TypeId_;
  std::unique_ptr<NodeAST> InitExpr_;

public:

  VarDeclAST(const SourceLocation & Loc, const std::vector<Identifier> & Vars, 
      Identifier VarType, std::unique_ptr<NodeAST> Init)
    : DeclAST(Loc), VarIds_(Vars), TypeId_(VarType),
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
class ArrayDeclAST : public VarDeclAST {
protected:

  std::unique_ptr<NodeAST> SizeExpr_;

public:

  ArrayDeclAST(const SourceLocation & Loc, const std::vector<Identifier> & VarNames, 
      Identifier VarType, std::unique_ptr<NodeAST> Init,
      std::unique_ptr<NodeAST> Size)
    : VarDeclAST(Loc, VarNames, VarType, std::move(Init)),
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

  Identifier Id_;
  std::unique_ptr<Identifier> ReturnTypeId_;
  bool IsOperator_ = false;
  unsigned Precedence_ = 0;  // Precedence if a binary op.
  
  std::vector<Identifier> ArgIds_;
  std::vector<Identifier> ArgTypeIds_;
  std::vector<bool> ArgIsArray_;

public:
  
  PrototypeAST(const Identifier & Id) : NodeAST(Id.getLoc()), Id_(Id)  {}

  PrototypeAST(
    const Identifier & Id,
    std::vector<Identifier> && Args,
    std::vector<Identifier> && ArgTypes,
    std::vector<bool> && ArgIsArray,
    std::unique_ptr<Identifier> Return,
    bool IsOperator = false,
    unsigned Prec = 0)
      : NodeAST(Id.getLoc()), Id_(Id), ReturnTypeId_(std::move(Return)),
        IsOperator_(IsOperator), Precedence_(Prec), ArgIds_(std::move(Args)),
        ArgTypeIds_(std::move(ArgTypes)), ArgIsArray_(std::move(ArgIsArray))
  {}

  
  virtual void accept(AstDispatcher& dispatcher) override;
  
  const std::string &getName() const { return Id_.getName(); }

  bool isUnaryOp() const { return IsOperator_ && ArgIds_.size() == 1; }
  bool isBinaryOp() const { return IsOperator_ && ArgIds_.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    auto Name = Id_.getName();
    return Name[Name.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence_; }
  auto getLoc() const { return Id_.getLoc(); }
  
  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;
};

////////////////////////////////////////////////////////////////////////////////
/// FunctionAST - This class represents a function definition itself.
////////////////////////////////////////////////////////////////////////////////
class FunctionAST : public NodeAST {
protected:

  std::unique_ptr<PrototypeAST> ProtoExpr_;
  ASTBlock BodyExprs_;
  std::unique_ptr<NodeAST> ReturnExpr_;

public:

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ASTBlock Body)
      : NodeAST(Proto->getLoc()), ProtoExpr_(std::move(Proto)),
        BodyExprs_(std::move(Body)) {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ASTBlock Body, 
      std::unique_ptr<NodeAST> Return)
      : NodeAST(Proto->getLoc()), ProtoExpr_(std::move(Proto)),
        BodyExprs_(std::move(Body)), ReturnExpr_(std::move(Return))
  {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<NodeAST> Return)
      : NodeAST(Proto->getLoc()), ProtoExpr_(std::move(Proto)),
      ReturnExpr_(std::move(Return))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Analyzer;
  friend class CodeGen;
  friend class Vizualizer;

};

} // namespace

#endif // CONTRA_AST_HPP
