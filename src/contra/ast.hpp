#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "visiter.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "identifier.hpp"
#include "sourceloc.hpp"
#include "symbols.hpp"

#include <cassert>
#include <deque>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// NodeAST - Base class for all nodes.
////////////////////////////////////////////////////////////////////////////////
class NodeAST {
  
  SourceLocation Loc_;

  FunctionDef* ParentFunction_ = nullptr;

public:

	NodeAST() = default;
  
  NodeAST(const SourceLocation & Loc) : Loc_(Loc) {}
  
  virtual ~NodeAST() = default;

  virtual void accept(AstVisiter& visiter) = 0;

  virtual std::string getClassName() const = 0;
  
  const auto & getLoc() const { return Loc_; }
  int getLine() const { return Loc_.getLine(); }
  int getCol() const { return Loc_.getCol(); }

  void setParentFunctionDef(FunctionDef* FunDef)
  { ParentFunction_ = FunDef; }

  FunctionDef* getParentFunctionDef() { return ParentFunction_; }

  virtual void setFuture(bool=true) {}
  virtual bool isFuture() const { return false; }
};

// some useful types
using ASTBlock = std::deque< std::unique_ptr<NodeAST> >;
using ASTBlockList = std::list<ASTBlock>;

inline auto createBlock( ASTBlockList & list)
{ return list.emplace( list.end(), ASTBlock{} ); }


////////////////////////////////////////////////////////////////////////////////
/// ExprAST - Base class for all expression nodes.
////////////////////////////////////////////////////////////////////////////////
class ExprAST : public NodeAST {
protected:

  VariableType Type_;
  
public:
  
  ExprAST(const SourceLocation & Loc) : NodeAST(Loc) {}
  ExprAST(const SourceLocation & Loc, VariableType Type) : NodeAST(Loc),
    Type_(Type) {}

  virtual ~ExprAST() = default;
  
  void setType(const VariableType & Type) { Type_ = Type; }
  const auto & getType() const { return Type_; }
  auto & getType() { return Type_; }


  virtual void setFuture(bool IsFuture=true) { Type_.setFuture(IsFuture); }
  virtual bool isFuture() const { return Type_.isFuture(); }

};

//==============================================================================
/// ValueExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
class ValueExprAST : public ExprAST {

public:

  enum class ValueType {
    Int, Real, String
  };

protected:

  std::string Val_;
  ValueType ValueType_;

public:
  ValueExprAST(
      const SourceLocation & Loc,
      const std::string & Val,
      ValueType Ty) :
    ExprAST(Loc),
    Val_(Val),
    ValueType_(Ty)
  {}

  template<typename T>
  T getVal() const;
  
  void accept(AstVisiter& visiter) override
  { visiter.visit(*this); }
  
  virtual std::string getClassName() const override
  { return "ValueExprAST"; }

  ValueType getValueType() const { return ValueType_; }
  
};


//==============================================================================
/// VarAccessExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
class VarAccessExprAST : public ExprAST {
protected:

  std::string Name_;
  VariableDef* VarDef_ = nullptr;

  // Note: Derived member VarType_ might differ from VarDef->getType().
  // This is because the accessed type might be different from the original
  // declaration.  An example is accessing a future variable as a non-future.

public:

  VarAccessExprAST(
      const SourceLocation & Loc,
      const std::string &Name) :
    ExprAST(Loc),
    Name_(Name)
  {}

  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "VarAccessExprAST"; };

  const std::string &getName() const { return Name_; }
  
  void setVariableDef(VariableDef* VarDef) { VarDef_=VarDef; }
  VariableDef* getVariableDef() const { return VarDef_; }
};

//==============================================================================
/// ArrayExprAST - Expression class for referencing an array.
//==============================================================================
class ArrayAccessExprAST : public VarAccessExprAST {
protected:
  
  std::unique_ptr<NodeAST> IndexExpr_;

public:
  ArrayAccessExprAST(
      const SourceLocation & Loc,
      const std::string &Name, 
      std::unique_ptr<NodeAST> IndexExpr) :
    VarAccessExprAST(Loc, Name),
    IndexExpr_(std::move(IndexExpr))
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "VarAccessExprAST"; };
  
  auto getIndexExpr() const { return IndexExpr_.get(); }

};

//==============================================================================
/// ArrayExprAST - Expression class for referencing an array.
//==============================================================================
class ArrayExprAST : public ExprAST {
protected:

  ASTBlock ValExprs_;
  std::unique_ptr<NodeAST> SizeExpr_;

public:

  ArrayExprAST(
      const SourceLocation & Loc,
      ASTBlock Vals,
      std::unique_ptr<NodeAST> Size) :
    ExprAST(Loc),
    ValExprs_(std::move(Vals)),
    SizeExpr_(std::move(Size))
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "ArrayExprAST"; };

  bool hasSize() const { return static_cast<bool>(SizeExpr_); }
  auto getSizeExpr() const { return SizeExpr_.get(); }

  auto getNumVals() const { return ValExprs_.size(); }
  auto getValExpr(int i) const { return ValExprs_[i].get(); }
  const auto & getValExprs() const { return ValExprs_; }

  auto moveValExpr(int i) { return std::move(ValExprs_[i]); }
  auto setValExpr(int i, std::unique_ptr<NodeAST> Expr) { ValExprs_[i] = std::move(Expr); }

};

//==============================================================================
/// RangeExprAST - Expression class for referencing  range.
//==============================================================================
class RangeExprAST : public ExprAST {
protected:

  std::unique_ptr<NodeAST> StartExpr_, EndExpr_;

public:

  RangeExprAST(
      const SourceLocation & Loc,
      std::unique_ptr<NodeAST> Start,
      std::unique_ptr<NodeAST> End) :
    ExprAST(Loc),
    StartExpr_(std::move(Start)),
    EndExpr_(std::move(End))
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "RangeExprAST"; };

  auto getStartExpr() const { return StartExpr_.get(); }
  auto moveStartExpr() { return std::move(StartExpr_); }
  auto setStartExpr(std::unique_ptr<NodeAST> Expr) { EndExpr_ = std::move(Expr); }

  auto getEndExpr() const { return EndExpr_.get(); }
  auto moveEndExpr() { return std::move(EndExpr_); }
  auto setEndExpr(std::unique_ptr<NodeAST> Expr) { EndExpr_ = std::move(Expr); }

};

//==============================================================================
/// CastExprAST - Expression class for casts
//==============================================================================
class CastExprAST : public ExprAST {
protected:

  std::unique_ptr<NodeAST> FromExpr_;
  Identifier TypeId_;


public:
  CastExprAST(
      const SourceLocation & Loc,
      std::unique_ptr<NodeAST> FromExpr,
      Identifier TypeId) :
    ExprAST(Loc),
    FromExpr_(std::move(FromExpr)),
    TypeId_(TypeId)
  {}

  CastExprAST(const SourceLocation & Loc, std::unique_ptr<NodeAST> FromExpr,
      const VariableType & Type) : ExprAST(Loc, Type), FromExpr_(std::move(FromExpr))
  {}

  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "CastExprAST"; };

  const auto & getTypeId() const { return TypeId_; }
  auto getFromExpr() const { return FromExpr_.get(); }
  
};


//==============================================================================
/// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
class UnaryExprAST : public ExprAST {
protected:

  char OpCode_;
  std::unique_ptr<NodeAST> OpExpr_;

public:
  UnaryExprAST(
      const SourceLocation & Loc,
      char Opcode,
      std::unique_ptr<NodeAST> Operand) :
    ExprAST(Loc),
    OpCode_(Opcode),
    OpExpr_(std::move(Operand))
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "UnaryExprAST"; };

  auto getOperand() const { return OpCode_; }
  auto getOpExpr() const { return OpExpr_.get(); }
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
  BinaryExprAST(
      const SourceLocation & Loc, 
      char Op, std::unique_ptr<NodeAST> lhs,
      std::unique_ptr<NodeAST> rhs) :
    ExprAST(Loc),
    OpCode_(Op),
    LeftExpr_(std::move(lhs)),
    RightExpr_(std::move(rhs))
  {}

  auto getOperand() const { return OpCode_; }
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "BinaryExprAST"; };

  auto getLeftExpr() const { return LeftExpr_.get(); }
  auto moveLeftExpr() { return std::move(LeftExpr_); }
  auto setLeftExpr(std::unique_ptr<NodeAST> Expr) { LeftExpr_ = std::move(Expr); }

  auto getRightExpr() const { return RightExpr_.get(); }
  auto moveRightExpr() { return std::move(RightExpr_); }
  auto setRightExpr(std::unique_ptr<NodeAST> Expr) { RightExpr_ = std::move(Expr); }
};

//==============================================================================
/// CallExprAST - Expression class for function calls.
//==============================================================================
class CallExprAST : public ExprAST {
protected:
  
  std::string Callee_;
  ASTBlock ArgExprs_;
  bool IsTopTask_ = false;
  std::vector<VariableType> ArgTypes_;
  
  FunctionDef* FunctionDef_ = nullptr;

public:

  CallExprAST(
      const SourceLocation & Loc,
      const std::string &Callee,
      ASTBlock Args) :
    ExprAST(Loc),
    Callee_(Callee),
    ArgExprs_(std::move(Args))
  {}

  const std::string & getName() const { return Callee_; }

  void setTopLevelTask(bool TopTask = true) { IsTopTask_ = TopTask; }
  bool isTopLevelTask() { return IsTopTask_; }
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "CallExprAST"; };

  auto getNumArgs() const { return ArgExprs_.size(); }
  const auto & getArgExprs() const { return ArgExprs_; }
  auto getArgExpr(int i) const { return ArgExprs_[i].get(); }
  
  auto moveArgExpr(int i) { return std::move(ArgExprs_[i]); }
  auto setArgExpr(int i, std::unique_ptr<NodeAST> Expr) { ArgExprs_[i] = std::move(Expr); }

  const auto & getArgType(int i) { return ArgTypes_[i]; }
  void setArgTypes(const std::vector<VariableType> & ArgTypes)
  { ArgTypes_ = ArgTypes; }

  auto getFunctionDef() const { return FunctionDef_; }
  void setFunctionDef(FunctionDef* F) { FunctionDef_ = F; }
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

  IfStmtAST(
      const SourceLocation & Loc,
      std::unique_ptr<NodeAST> Cond,
      ASTBlock Then) :
    StmtAST(Loc),
    CondExpr_(std::move(Cond)),
    ThenExpr_(std::move(Then))
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "IfStmtAST"; };

  static std::unique_ptr<NodeAST> makeNested( 
    std::list< std::pair<SourceLocation, std::unique_ptr<NodeAST>> > & Conds,
    ASTBlockList & Blocks );

  auto getCondExpr() const { return CondExpr_.get(); }
  const auto & getThenExprs() const { return ThenExpr_; }
  const auto & getElseExprs() const { return ElseExpr_; }
};

//==============================================================================
// ForExprAST - Expression class for for/in.
//==============================================================================
class ForStmtAST : public StmtAST {

public:

  enum class LoopType {
    To, Until, Range
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
      LoopType Loop = LoopType::To) :
    StmtAST(Loc),
    VarId_(VarId),
    StartExpr_(std::move(Start)),
    EndExpr_(std::move(End)),
    StepExpr_(std::move(Step)),
    BodyExprs_(std::move(Body)),
    Loop_(Loop)
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "ForStmtAST"; };

  const std::string & getVarName() const { return VarId_.getName(); }

  const auto & getVarId() const { return VarId_; }

  auto getLoopType() const { return Loop_; }
  
  const auto & getBodyExprs() const { return BodyExprs_; }

  auto getStartExpr() const { return StartExpr_.get(); }

  auto hasEnd() const { return static_cast<bool>(EndExpr_); }
  auto getEndExpr() const { return EndExpr_.get(); }

  auto hasStep() const { return static_cast<bool>(StepExpr_); }
  auto getStepExpr() const { return StepExpr_.get(); }
};

//==============================================================================
// ForEachExprAST - Expression class for for/in.
//==============================================================================
class ForeachStmtAST : public ForStmtAST {

  std::vector<VariableDef*> AccessedVariables_;
  std::string Name_;
  bool IsLifted_ = false;
  unsigned NumParts_ = 0;

public:
  
  ForeachStmtAST(
      const SourceLocation & Loc,
      const Identifier &VarId,
      std::unique_ptr<NodeAST> Start,
      std::unique_ptr<NodeAST> End,
      std::unique_ptr<NodeAST> Step,
      ASTBlock Body,
      LoopType Loop = LoopType::To) :
    ForStmtAST(
        Loc,
        VarId,
        std::move(Start),
        std::move(End),
        std::move(Step),
        std::move(Body),
        Loop)
  {}
  
  virtual void accept(AstVisiter& visiter) override;

  void setAccessedVariables(const std::vector<VariableDef*> & VarDefs)
  { AccessedVariables_ = VarDefs; }

  const auto & getAccessedVariables()
  { return AccessedVariables_; }

  auto getBodyExpr(unsigned i) const { return BodyExprs_[i].get(); }

  auto moveBodyExprs() {
    ASTBlock NewBody;
    auto NumBody = BodyExprs_.size();
    for (unsigned i=NumParts_; i<NumBody; ++i) {
      NewBody.emplace_front( std::move(BodyExprs_.back()) );
      BodyExprs_.pop_back();
    }
    return NewBody;
  }
  
  const std::string &getName() const { return Name_; }
  void setName(const std::string& Name) { Name_ = Name; }
  
  bool isLifted() const { return IsLifted_; }
  void setLifted(bool IsLifted=true) { IsLifted_ = IsLifted; }
  
  void setNumPartitions(unsigned NumParts) { NumParts_=NumParts; }
  auto getNumPartitions() const { return NumParts_; }
};

//==============================================================================
/// Partition statement
//==============================================================================
class PartitionStmtAST : public StmtAST {
protected:

  Identifier RangeId_;
  std::unique_ptr<NodeAST> ColorExpr_;
  ASTBlock BodyExprs_;
  std::vector<VariableDef*> AccessedVariables_;

public:
  PartitionStmtAST(
      const SourceLocation & Loc,
      const Identifier & RangeId,
      std::unique_ptr<NodeAST> ColorExpr,
      ASTBlock BodyExprs) :
    StmtAST(Loc),
    RangeId_(RangeId),
    ColorExpr_(std::move(ColorExpr)),
    BodyExprs_(std::move(BodyExprs))
  {}

  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "PartitionStmtAST"; };

  const auto & getVarName() const { return RangeId_.getName(); }
  const auto & getVarId() const { return RangeId_; }

  auto getColorExpr() const { return ColorExpr_.get(); }
  
  bool hasBodyExprs() const { return !BodyExprs_.empty(); }
  const auto & getBodyExprs() const { return BodyExprs_; }
  
  void setAccessedVariables(const std::vector<VariableDef*> & VarDefs)
  { AccessedVariables_ = VarDefs; }

  const auto & getAccessedVariables()
  { return AccessedVariables_; }

};



//==============================================================================
/// Assignment statement
//==============================================================================
class AssignStmtAST : public StmtAST {
protected:

  std::unique_ptr<NodeAST> LeftExpr_;
  std::unique_ptr<NodeAST> RightExpr_;

public:
  AssignStmtAST(
      const SourceLocation & Loc,
      std::unique_ptr<NodeAST> lhs,
      std::unique_ptr<NodeAST> rhs) :
    StmtAST(Loc),
    LeftExpr_(std::move(lhs)),
    RightExpr_(std::move(rhs))
  {}

  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "AssignStmtAST"; };

  auto getLeftExpr() const { return LeftExpr_.get(); }
  auto moveLeftExpr() { return std::move(LeftExpr_); }
  auto setLeftExpr(std::unique_ptr<NodeAST> Expr) { LeftExpr_ = std::move(Expr); }

  auto getRightExpr() const { return RightExpr_.get(); }
  auto moveRightExpr() { return std::move(RightExpr_); }
  auto setRightExpr(std::unique_ptr<NodeAST> Expr) { RightExpr_ = std::move(Expr); }
};



//==============================================================================
/// VarDefExprAST - Expression class for var/in
//==============================================================================
class VarDeclAST : public StmtAST {
public:

  enum class AttrType {
    None, Array, Range
  };

protected:

  std::vector<Identifier> VarIds_;
  Identifier TypeId_;
  std::unique_ptr<NodeAST> InitExpr_;
  std::unique_ptr<NodeAST> SizeExpr_;
  std::unique_ptr<NodeAST> IndexExpr_;
  AttrType Attr_ = AttrType::None;


  std::vector<VariableDef*> VarDefs_;

public:

  VarDeclAST(
      const SourceLocation & Loc,
      const std::vector<Identifier> & Vars, 
      Identifier VarType,
      std::unique_ptr<NodeAST> Init,
      std::unique_ptr<NodeAST> Size,
      AttrType Attr = AttrType::None) :
    StmtAST(Loc),
    VarIds_(Vars),
    TypeId_(VarType),
    InitExpr_(std::move(Init)),
    SizeExpr_(std::move(Size)),
    Attr_(Attr),
    VarDefs_(Vars.size(),nullptr)
  {}

  bool isArray() const { return Attr_ == AttrType::Array; }
  void setArray(bool IsArray=true) {
    if (Attr_ != AttrType::Array && IsArray) Attr_ = AttrType::Array;
    if (Attr_ == AttrType::Array && !IsArray) Attr_ = AttrType::None;
  }
  
  bool isRange() const { return Attr_ == AttrType::Range; }
  void setRange(bool IsRange=true) {
    if (Attr_ != AttrType::Range && IsRange) Attr_ = AttrType::Range;
    if (Attr_ == AttrType::Range && !IsRange) Attr_ = AttrType::None;
  }
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "VarDeclAST"; };

  std::vector<std::string> getVarNames() const
  {
    std::vector<std::string> strs;
    for (const auto & Id : VarIds_)
      strs.emplace_back( Id.getName() );
    return strs;
  }

  const auto & getTypeId() const { return TypeId_; }

  auto getNumVars() const { return VarIds_.size(); }
  const auto & getVarIds() const { return VarIds_; }
  const auto & getVarId(unsigned i) const { return VarIds_[i]; }
  const auto & getVarName(unsigned i) const { return VarIds_[i].getName(); }

  const auto & getVarType(unsigned i) const { return VarDefs_[i]->getType(); }
  auto & getVarType(unsigned i) { return VarDefs_[i]->getType(); }

  auto getInitExpr() const { return InitExpr_.get(); }
  auto moveInitExpr() { return std::move(InitExpr_); }
  auto setInitExpr(std::unique_ptr<NodeAST> Init) { InitExpr_ = std::move(Init); }
  
  bool hasSize() const { return static_cast<bool>(SizeExpr_); }
  auto getSizeExpr() const { return SizeExpr_.get(); }

  void setVariableDef(unsigned i, VariableDef* VarDef) { VarDefs_[i]=VarDef; }
  VariableDef* getVariableDef(unsigned i) const { return VarDefs_[i]; }
  
  bool isFuture(unsigned i) const { return VarDefs_[i]->getType().isFuture(); }
};

//==============================================================================
/// VarDefExprAST - Expression class for var/in
//==============================================================================
class FieldDeclAST : public VarDeclAST {

  std::unique_ptr<NodeAST> PartitionExpr_;

public:

  FieldDeclAST(
      const SourceLocation & Loc,
      const std::vector<Identifier> & Vars, 
      Identifier VarType,
      std::unique_ptr<NodeAST> Init,
      std::unique_ptr<NodeAST> Size,
      std::unique_ptr<NodeAST> Part) :
    VarDeclAST(
        Loc,
        Vars,
        VarType,
        std::move(Init),
        std::move(Size),
        (static_cast<bool>(Size)) ? AttrType::Array : AttrType::None),
    PartitionExpr_(std::move(Part))
  {}
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "FieldDeclAST"; };

  auto getPartExpr() const { return PartitionExpr_.get(); }

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
  
  std::vector<VariableType> ArgTypes_;
  VariableType ReturnType_;

  bool IsAnonExpr_ = false;

public:
  
  PrototypeAST(const Identifier & Id) :
    NodeAST(Id.getLoc()), Id_(Id), IsAnonExpr_(true)
  {}

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

  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "PrototypeAST"; };
  
  const std::string &getName() const { return Id_.getName(); }
  const auto & getId() const { return Id_; }

  bool isUnaryOp() const { return IsOperator_ && ArgIds_.size() == 1; }
  bool isBinaryOp() const { return IsOperator_ && ArgIds_.size() == 2; }
  bool isAnonExpr() const { return IsAnonExpr_; } 

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    auto Name = Id_.getName();
    return Name[Name.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence_; }
  auto getLoc() const { return Id_.getLoc(); }

  const auto & getReturnType() const { return ReturnType_; }
  void setReturnType(const VariableType & ReturnType) { ReturnType_ = ReturnType; }

  auto hasReturn() const { return static_cast<bool>(ReturnTypeId_); }
  const auto & getReturnTypeId() const { return *ReturnTypeId_; }

  auto getNumArgs() const { return ArgIds_.size(); } 
  const auto & getArgTypeId(int i) const { return ArgTypeIds_[i]; }
  const auto & getArgId(int i) const { return ArgIds_[i]; }
  const auto & getArgName(int i) const { return ArgIds_[i].getName(); }
  
  auto isArgArray(int i) const { return ArgIsArray_[i]; } 

  const auto & getArgType(int i) { return ArgTypes_[i]; }

  void setArgTypes(const std::vector<VariableType> & ArgTypes)
  { ArgTypes_ = ArgTypes; }
  
};

////////////////////////////////////////////////////////////////////////////////
/// FunctionAST - This class represents a function definition itself.
////////////////////////////////////////////////////////////////////////////////
class FunctionAST : public NodeAST {
protected:

  std::unique_ptr<PrototypeAST> ProtoExpr_;
  ASTBlock BodyExprs_;
  std::unique_ptr<NodeAST> ReturnExpr_;
  bool IsTopExpression_ = false;
  bool IsTask_ = false;
  std::string Name_;

  FunctionDef* FunctionDef_ = nullptr;

public:
  
  FunctionAST(
      const std::string & Name,
      ASTBlock Body,
      bool IsTask = false) :
    BodyExprs_(std::move(Body)),
    IsTask_(IsTask),
    Name_(Name)
  {}

  FunctionAST(
      std::unique_ptr<PrototypeAST> Proto,
      ASTBlock Body, 
      std::unique_ptr<NodeAST> Return,
      bool IsTask = false) :
    NodeAST(Proto->getLoc()),
    ProtoExpr_(std::move(Proto)),
    BodyExprs_(std::move(Body)),
    ReturnExpr_(std::move(Return)),
    IsTask_(IsTask),
    Name_(ProtoExpr_->getName())
  {}

  FunctionAST(
      std::unique_ptr<PrototypeAST> Proto,
      std::unique_ptr<NodeAST> Return) :
    NodeAST(Proto->getLoc()),
    ProtoExpr_(std::move(Proto)),
    ReturnExpr_(std::move(Return)),
    IsTopExpression_(true),
    Name_(ProtoExpr_->getName())
  {}

  auto isTopLevelExpression() const { return IsTopExpression_; }
  auto isTask() const { return IsTask_; }
  const std::string &getName() const { return Name_; }
  
  virtual void accept(AstVisiter& visiter) override;

  virtual std::string getClassName() const override
  { return "FunctionAST"; };

  bool hasReturn() const { return static_cast<bool>(ReturnExpr_); }
  auto getReturnExpr() const { return ReturnExpr_.get(); }
  
  auto moveReturnExpr() { return std::move(ReturnExpr_); }
  auto setReturnExpr(std::unique_ptr<NodeAST> Expr) { ReturnExpr_ = std::move(Expr); }
  
  auto getProtoExpr() const { return ProtoExpr_.get(); }
  auto moveProtoExpr() { return std::move(ProtoExpr_); }
  
  auto getNumBodyExprs() const { return BodyExprs_.size(); }
  const auto & getBodyExprs() const { return BodyExprs_; }

  auto getFunctionDef() const { return FunctionDef_; }
  void setFunctionDef(FunctionDef* F) { FunctionDef_ = F; }
};

////////////////////////////////////////////////////////////////////////////////
/// TaskAST - This class represents a function definition itself.
////////////////////////////////////////////////////////////////////////////////
class TaskAST : public FunctionAST {

public:

  TaskAST(
      std::unique_ptr<PrototypeAST> Proto,
      ASTBlock Body, 
      std::unique_ptr<NodeAST> Return) :
    FunctionAST(
        std::move(Proto),
        std::move(Body),
        std::move(Return),
        true)
  {}

  virtual void accept(AstVisiter& visiter) override;

  virtual std::string getClassName() const override
  { return "TaskAST"; };

};

////////////////////////////////////////////////////////////////////////////////
/// TaskAST - This class represents a function definition itself.
////////////////////////////////////////////////////////////////////////////////
class IndexTaskAST : public FunctionAST {

  std::string LoopVarName_;
  std::vector<VariableDef*> Vars_;
  std::vector<bool> VarIsPartitioned_;

public:

  IndexTaskAST(
      const std::string & Name,
      ASTBlock Body,
      const std::string & LoopVar,
      const std::vector<VariableDef*>& Vars,
      const std::vector<bool>& VarIsPartitioned) :
    FunctionAST(Name, std::move(Body), true),
    LoopVarName_(LoopVar),
    Vars_(Vars),
    VarIsPartitioned_(VarIsPartitioned)
  {}

  virtual void accept(AstVisiter& visiter) override;

  virtual std::string getClassName() const override
  { return "IndexTaskAST"; };
 
  const auto & getVariableDefs()
  { return Vars_; }

  auto getVariableDef(unsigned i) const { return Vars_[i]; }
 
  const auto & getLoopVariableName() const { return LoopVarName_; }
  const auto & getName() const { return Name_; }

  const auto & getVarIsPartitioned() const { return VarIsPartitioned_; }

};

} // namespace

#endif // CONTRA_AST_HPP
