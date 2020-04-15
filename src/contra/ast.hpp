#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "visiter.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "identifier.hpp"
#include "sourceloc.hpp"
#include "symbols.hpp"

#include <cassert>
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

public:

	NodeAST() = default;
  
  NodeAST(const SourceLocation & Loc) : Loc_(Loc) {}
  
  virtual ~NodeAST() = default;

  virtual void accept(AstVisiter& visiter) = 0;

  virtual std::string getClassName() const = 0;
  
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
  ValueExprAST(const SourceLocation & Loc, const std::string & Val, ValueType Ty)
    : ExprAST(Loc), Val_(Val), ValueType_(Ty) {}

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
  VariableType Type_;
  bool NeedValue_ = true;

public:

  VarAccessExprAST(const SourceLocation & Loc, const std::string &Name)
    : ExprAST(Loc), Name_(Name)
  {}

  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "VarAccessExprAST"; };

  const std::string &getName() const { return Name_; }
  
  void setNeedValue(bool NeedValue=true) { NeedValue_=NeedValue; }
  bool needValue() const { return NeedValue_; }
};

//==============================================================================
/// ArrayExprAST - Expression class for referencing an array.
//==============================================================================
class ArrayAccessExprAST : public VarAccessExprAST {
protected:
  
  std::unique_ptr<NodeAST> IndexExpr_;

public:
  ArrayAccessExprAST(const SourceLocation & Loc, const std::string &Name,
      std::unique_ptr<NodeAST> IndexExpr)
    : VarAccessExprAST(Loc, Name), IndexExpr_(std::move(IndexExpr))
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

  ArrayExprAST(const SourceLocation & Loc, ASTBlock Vals,
      std::unique_ptr<NodeAST> Size)
    : ExprAST(Loc), ValExprs_(std::move(Vals)), SizeExpr_(std::move(Size))
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

  CastExprAST(const SourceLocation & Loc, std::unique_ptr<NodeAST> FromExpr,
      VariableType Type) : ExprAST(Loc, Type), FromExpr_(std::move(FromExpr))
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
  UnaryExprAST(const SourceLocation & Loc,
      char Opcode,
      std::unique_ptr<NodeAST> Operand)
    : ExprAST(Loc), OpCode_(Opcode), OpExpr_(std::move(Operand))
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
  BinaryExprAST(const SourceLocation & Loc, 
      char Op, std::unique_ptr<NodeAST> lhs,
      std::unique_ptr<NodeAST> rhs)
    : ExprAST(Loc), OpCode_(Op), LeftExpr_(std::move(lhs)), RightExpr_(std::move(rhs))
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

public:

  CallExprAST(const SourceLocation & Loc,
      const std::string &Callee,
      ASTBlock Args)
    : ExprAST(Loc), Callee_(Callee), ArgExprs_(std::move(Args))
  {}

  const std::string & getName() const { return Callee_; }

  void setTopLevelTask(bool TopTask = true) { IsTopTask_ = TopTask; }
  bool isTopLevelTask() { return IsTopTask_; }
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "CallExprAST"; };

  auto getNumArgs() const { return ArgExprs_.size(); }
  auto getArgExpr(int i) const { return ArgExprs_[i].get(); }
  
  auto moveArgExpr(int i) { return std::move(ArgExprs_[i]); }
  auto setArgExpr(int i, std::unique_ptr<NodeAST> Expr) { ArgExprs_[i] = std::move(Expr); }
  
  const auto & getArgType(int i) { return ArgTypes_[i]; }
  void setArgTypes(const std::vector<VariableType> & ArgTypes)
  { ArgTypes_ = ArgTypes; }

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
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "ForStmtAST"; };

  const std::string & getVarName() const { return VarId_.getName(); }

  const auto & getVarId() const { return VarId_; }

  auto getLoopType() const { return Loop_; }
  
  const auto & getBodyExprs() const { return BodyExprs_; }

  auto getStartExpr() const { return StartExpr_.get(); }
  auto getEndExpr() const { return EndExpr_.get(); }

  auto hasStep() const { return static_cast<bool>(StepExpr_); }
  auto getStepExpr() const { return StepExpr_.get(); }
};

//==============================================================================
// ForEachExprAST - Expression class for for/in.
//==============================================================================
class ForeachStmtAST : public ForStmtAST {

  std::string Name_;
  std::set<std::string> AccessedVariables_;

public:
  
  ForeachStmtAST(const SourceLocation & Loc,
      const Identifier &VarId,
      std::unique_ptr<NodeAST> Start,
      std::unique_ptr<NodeAST> End,
      std::unique_ptr<NodeAST> Step,
      ASTBlock Body,
      LoopType Loop = LoopType::To) :
    ForStmtAST(Loc, VarId, std::move(Start), std::move(End), std::move(Step),
        std::move(Body), Loop)
  {}
  
  virtual void accept(AstVisiter& visiter) override;

  void addAccessedVariable(const std::string & VarName)
  { AccessedVariables_.emplace(VarName); }

  const auto & getAccessedVariables()
  { return AccessedVariables_; }

  auto moveBodyExprs() { return std::move(BodyExprs_); }

  void setName(const std::string& Name) { Name_ = Name; }
  const auto & getName() const { return Name_; }
  
};

//==============================================================================
/// Assignment statement
//==============================================================================
class AssignStmtAST : public StmtAST {
protected:

  std::unique_ptr<NodeAST> LeftExpr_;
  std::unique_ptr<NodeAST> RightExpr_;

public:
  AssignStmtAST(const SourceLocation & Loc, std::unique_ptr<NodeAST> lhs,
      std::unique_ptr<NodeAST> rhs)
    : StmtAST(Loc), LeftExpr_(std::move(lhs)), RightExpr_(std::move(rhs))
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

protected:

  VariableType Type_;
  std::vector<Identifier> VarIds_;
  Identifier TypeId_;
  std::unique_ptr<NodeAST> InitExpr_;
  std::unique_ptr<NodeAST> SizeExpr_;
  bool IsArray_ = false;

public:

  VarDeclAST(const SourceLocation & Loc, const std::vector<Identifier> & Vars, 
      Identifier VarType, std::unique_ptr<NodeAST> Init)
    : StmtAST(Loc), VarIds_(Vars), TypeId_(VarType),
      InitExpr_(std::move(Init)), IsArray_(false)
  {}
  
  VarDeclAST(const SourceLocation & Loc, const std::vector<Identifier> & Vars, 
      Identifier VarType, std::unique_ptr<NodeAST> Init,
      std::unique_ptr<NodeAST> Size)
    : StmtAST(Loc), VarIds_(Vars), TypeId_(VarType),
      InitExpr_(std::move(Init)),
      SizeExpr_(std::move(Size)), IsArray_(true)
  {}

  void setType(const VariableType & Type) { Type_ = Type; }
  const auto & getType() const { return Type_; }

  bool isArray() const { return IsArray_; }
  void setArray(bool IsArray=true) { IsArray_ = IsArray; }
  
  virtual void accept(AstVisiter& visiter) override;
  
  virtual std::string getClassName() const override
  { return "VarDeclAST"; };

  std::vector<std::string> getNames() const
  {
    std::vector<std::string> strs;
    for (const auto & Id : VarIds_)
      strs.emplace_back( Id.getName() );
    return strs;
  }

  const auto & getTypeId() const { return TypeId_; }

  auto getNumVars() const { return VarIds_.size(); }
  const auto & getVarIds() const { return VarIds_; }
  const auto & getVarId(int i) const { return VarIds_[i]; }
  const auto & getVarName(int i) const { return VarIds_[i].getName(); }

  auto getInitExpr() const { return InitExpr_.get(); }
  auto moveInitExpr() { return std::move(InitExpr_); }
  auto setInitExpr(std::unique_ptr<NodeAST> Init) { InitExpr_ = std::move(Init); }
  
  bool hasSize() const { return static_cast<bool>(SizeExpr_); }
  auto getSizeExpr() const { return SizeExpr_.get(); }
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
  
  PrototypeAST(const Identifier & Id) : NodeAST(Id.getLoc()), Id_(Id), IsAnonExpr_(true)  {}

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

public:
  
FunctionAST(const std::string & Name, ASTBlock Body, bool IsTask = false)
      : BodyExprs_(std::move(Body)), IsTask_(IsTask), Name_(Name)
  {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ASTBlock Body, 
      std::unique_ptr<NodeAST> Return, bool IsTask = false)
      : NodeAST(Proto->getLoc()), ProtoExpr_(std::move(Proto)),
        BodyExprs_(std::move(Body)), ReturnExpr_(std::move(Return)),
        IsTask_(IsTask), Name_(ProtoExpr_->getName())
  {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<NodeAST> Return)
      : NodeAST(Proto->getLoc()), ProtoExpr_(std::move(Proto)),
        ReturnExpr_(std::move(Return)), IsTopExpression_(true),
        Name_(ProtoExpr_->getName())
  {}

  auto isTopLevelExpression() const { return IsTopExpression_; }
  auto isTask() const { return IsTask_; }
  const std::string &getName() const { return Name_; }
  
  virtual void accept(AstVisiter& visiter) override;

  virtual std::string getClassName() const override
  { return "FunctionAST"; };

  auto getReturnExpr() const { return ReturnExpr_.get(); }
  
  auto getProtoExpr() const { return ProtoExpr_.get(); }
  auto moveProtoExpr() { return std::move(ProtoExpr_); }
  
  auto getNumBodyExprs() const { return BodyExprs_.size(); }
  const auto & getBodyExprs() const { return BodyExprs_; }
};

////////////////////////////////////////////////////////////////////////////////
/// TaskAST - This class represents a function definition itself.
////////////////////////////////////////////////////////////////////////////////
class TaskAST : public FunctionAST {

public:

  TaskAST(std::unique_ptr<PrototypeAST> Proto, ASTBlock Body, 
      std::unique_ptr<NodeAST> Return)
      : FunctionAST(std::move(Proto), std::move(Body), std::move(Return), true)
  {}

  virtual void accept(AstVisiter& visiter) override;

  virtual std::string getClassName() const override
  { return "TaskAST"; };

};

////////////////////////////////////////////////////////////////////////////////
/// TaskAST - This class represents a function definition itself.
////////////////////////////////////////////////////////////////////////////////
class IndexTaskAST : public FunctionAST {

	using VariableEntry = std::shared_ptr<VariableDef>;
  std::string LoopVarName_;
  std::vector<VariableEntry> Vars_;

public:

  IndexTaskAST(const std::string & Name, ASTBlock Body,
      const std::string & LoopVar, const std::vector<VariableEntry>& Vars)
      : FunctionAST(Name, std::move(Body), true), LoopVarName_(LoopVar),
 				Vars_(Vars)
  {}

  virtual void accept(AstVisiter& visiter) override;

  virtual std::string getClassName() const override
  { return "IndexTaskAST"; };
 
  const auto & getVariables()
  { return Vars_; }
 
  const auto & getLoopVariableName() const { return LoopVarName_; }
  const auto & getName() const { return Name_; }

};

} // namespace

#endif // CONTRA_AST_HPP
