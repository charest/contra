#ifndef CONTRA_AST_HPP
#define CONTRA_AST_HPP

#include "codegen.hpp"
#include "config.hpp"
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
  
  VarTypes InferredType;

  ExprAST(SourceLocation Loc, VarTypes Type = VarTypes::Void)
    : Loc_(Loc), InferredType(Type) {}

  virtual ~ExprAST() = default;
  
  auto getLoc() const { return Loc_; }
  int getLine() const { return Loc_.getLine(); }
  int getCol() const { return Loc_.getCol(); }

};

//==============================================================================
/// ExprBlock - List of expressions that form a block 
//==============================================================================

struct ExprLocPair {
  SourceLocation Loc;
  std::unique_ptr<ExprAST> Expr;
};
using ExprLocPairList = std::list< ExprLocPair >;

inline
void addExpr(ExprLocPairList & l, SourceLocation sl, std::unique_ptr<ExprAST> e)
{ l.emplace_back( ExprLocPair{sl, std::move(e) } ); }


using ExprBlock = std::vector< std::unique_ptr<ExprAST> >;
using ExprBlockList = std::list<ExprBlock>;

inline auto createBlock( ExprBlockList & list)
{ return list.emplace( list.end(), ExprBlock{} ); }

//==============================================================================
/// ValueExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
template< typename T >
class ValueExprAST : public ExprAST {
protected:

  T Val_;

public:
  ValueExprAST(SourceLocation Loc, T Val)
    : ExprAST(Loc, getVarType<T>()), Val_(Val) {}

  const T & getVal() const { return Val_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;
  
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
  std::shared_ptr<ExprAST> Index_;

public:

  VariableExprAST(SourceLocation Loc, const std::string &Name,
      VarTypes Type) : ExprAST(Loc, Type), Name_(Name)
  {}

  VariableExprAST(SourceLocation Loc, const std::string &Name,
      VarTypes Type, std::unique_ptr<ExprAST> Index)
    : ExprAST(Loc, Type), Name_(Name), Index_(std::move(Index))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  const std::string &getName() const { return Name_; }
  
  bool isArray() const { return static_cast<bool>(Index_); }

  std::shared_ptr<ExprAST> getIndex() const { return Index_; }
  
  friend class Vizualizer;
  friend class CodeGen;
};

//==============================================================================
/// ArrayExprAST - Expression class for referencing an array.
//==============================================================================
class ArrayExprAST : public ExprAST {
protected:

  ExprBlock Vals_;
  std::unique_ptr<ExprAST> Size_;

public:

  ArrayExprAST(SourceLocation Loc, VarTypes VarType, ExprBlock Vals,
      std::unique_ptr<ExprAST> Size)
    : ExprAST(Loc, VarType), Vals_(std::move(Vals)), Size_(std::move(Size))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;

};

//==============================================================================
/// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
class BinaryExprAST : public ExprAST {
protected:

  char Op_;
  std::shared_ptr<ExprAST> LHS_;
  std::unique_ptr<ExprAST> RHS_;

public:
  BinaryExprAST(SourceLocation Loc, 
      char Op, std::unique_ptr<ExprAST> lhs,
      std::unique_ptr<ExprAST> rhs)
    : ExprAST(Loc), Op_(Op), LHS_(std::move(lhs)), RHS_(std::move(rhs))
  {
    // promote lesser types
    if (LHS_->InferredType == VarTypes::Real || RHS_->InferredType == VarTypes::Real)
      InferredType = VarTypes::Real;
    else if (LHS_->InferredType != RHS_->InferredType)
      THROW_SYNTAX_ERROR( "Don't know how to handle binary expression with '"
          << getVarTypeName(LHS_->InferredType) << "' and '"
          << getVarTypeName(RHS_->InferredType) << "'", Loc.getLine() );
    else
      InferredType =  LHS_->InferredType;
  }

  char getOperand() const { return Op_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;
};

//==============================================================================
/// CallExprAST - Expression class for function calls.
//==============================================================================
class CallExprAST : public ExprAST {
protected:
  
  std::string Callee_;
  std::vector<std::unique_ptr<ExprAST>> Args_;

public:

  CallExprAST(SourceLocation Loc,
      const std::string &Callee,
      std::vector<std::unique_ptr<ExprAST>> Args)
    : ExprAST(Loc), Callee_(Callee), Args_(std::move(Args))
  {}

  const std::string & getCalleeName() const { return Callee_; }
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;
  
};

//==============================================================================
/// IfExprAST - Expression class for if/then/else.
//==============================================================================
class IfExprAST : public ExprAST {
protected:

  std::unique_ptr<ExprAST> Cond_;
  ExprBlock Then_, Else_;

public:

  IfExprAST(SourceLocation Loc, std::unique_ptr<ExprAST> Cond,
       ExprBlock Then)
    : ExprAST(Loc), Cond_(std::move(Cond)), Then_(std::move(Then))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  static std::unique_ptr<ExprAST> makeNested( 
    ExprLocPairList & Conds, ExprBlockList & Blocks );
  
  friend class Vizualizer;
  friend class CodeGen;
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

  std::string VarName_;
  std::unique_ptr<ExprAST> Start_, End_, Step_;
  ExprBlock Body_;
  LoopType Loop_;

public:

  ForExprAST(SourceLocation Loc,
      const std::string &VarName,
      std::unique_ptr<ExprAST> Start,
      std::unique_ptr<ExprAST> End,
      std::unique_ptr<ExprAST> Step,
      ExprBlock Body,
      LoopType Loop = LoopType::To)
    : ExprAST(Loc), VarName_(VarName), Start_(std::move(Start)),
      End_(std::move(End)), Step_(std::move(Step)), Body_(std::move(Body)),
      Loop_(Loop)
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;
};

//==============================================================================
/// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
class UnaryExprAST : public ExprAST {
protected:

  char Opcode_;
  std::unique_ptr<ExprAST> Operand_;

public:
  UnaryExprAST(SourceLocation Loc,
      char Opcode,
      std::unique_ptr<ExprAST> Operand)
    : ExprAST(Loc, Operand->InferredType), Opcode_(Opcode), Operand_(std::move(Operand))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;
};

//==============================================================================
/// VarExprAST - Expression class for var/in
//==============================================================================
class VarExprAST : public ExprAST {

protected:

  std::vector<std::string> VarNames_;
  VarTypes VarType_;
  std::shared_ptr<ExprAST> Init_;

public:

  VarExprAST(SourceLocation Loc, const std::vector<std::string> & VarNames, 
      VarTypes VarType, std::unique_ptr<ExprAST> Init)
    : ExprAST(Loc, Init->InferredType), VarNames_(VarNames), VarType_(VarType),
      Init_(std::move(Init)) 
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;
 
  friend class Vizualizer;
  friend class CodeGen;
};

//==============================================================================
/// ArrayVarExprAST - Expression class for var/in
//==============================================================================
class ArrayVarExprAST : public VarExprAST {
protected:

  std::unique_ptr<ExprAST> Size_;

public:

  ArrayVarExprAST(SourceLocation Loc, const std::vector<std::string> & VarNames, 
      VarTypes VarType, std::unique_ptr<ExprAST> Init,
      std::unique_ptr<ExprAST> Size)
    : VarExprAST(Loc, VarNames, VarType, std::move(Init)),
      Size_(std::move(Size))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;
  
};

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
//==============================================================================
class PrototypeAST : public NodeAST {
protected:

  std::string Name_;
  VarTypes Return_;
  bool IsOperator_ = false;
  unsigned Precedence_ = 0;  // Precedence if a binary op.
  int Line_;
  
  std::vector< std::pair<std::string, Symbol> > Args_;

public:
  
  PrototypeAST(
    SourceLocation Loc,
    const std::string &Name,
    VarTypes Return)
      : Name_(Name), Return_(Return), Line_(Loc.getLine())
  {}

  PrototypeAST(
    SourceLocation Loc,
    const std::string &Name,
    std::vector< std::pair<std::string, Symbol> > && Args,
    VarTypes Return,
    bool IsOperator = false,
    unsigned Prec = 0)
      : Name_(Name), Return_(Return), IsOperator_(IsOperator),
        Precedence_(Prec), Line_(Loc.getLine()), Args_(std::move(Args))
  {}

  
  virtual void accept(AstDispatcher& dispatcher) override;
  
  const std::string &getName() const { return Name_; }

  bool isUnaryOp() const { return IsOperator_ && Args_.size() == 1; }
  bool isBinaryOp() const { return IsOperator_ && Args_.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name_[Name_.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence_; }
  int getLine() const { return Line_; }

  Symbol getArgSymbol(int i) { return Args_[i].second; }
  
  friend class Vizualizer;
  friend class CodeGen;
};

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
class FunctionAST : public NodeAST {
protected:

  std::unique_ptr<PrototypeAST> Proto_;
  ExprBlock Body_;
  std::unique_ptr<ExprAST> Return_;

public:

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ExprBlock Body)
      : Proto_(std::move(Proto)), Body_(std::move(Body)) {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, ExprBlock Body, 
      std::unique_ptr<ExprAST> Return)
      : Proto_(std::move(Proto)), Body_(std::move(Body)), Return_(std::move(Return))
  {}

  FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Return)
      : Proto_(std::move(Proto)), Return_(std::move(Return))
  {}
  
  virtual void accept(AstDispatcher& dispatcher) override;

  friend class Vizualizer;
  friend class CodeGen;

};

} // namespace

#endif // CONTRA_AST_HPP
