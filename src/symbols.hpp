#ifndef CONTRA_SYMBOLS_HPP
#define CONTRA_SYMBOLS_HPP

#include "sourceloc.hpp"
#include "vartype.hpp"

#include <map>
#include <string>

namespace contra {
  
//==============================================================================
// The base symbol type
//==============================================================================
class Symbol {

  std::string Name_;
  SourceLocation Loc_;

public:

  Symbol(const std::string & Name, SourceLocation Loc) : Name_(Name), Loc_(Loc)
  {}

  virtual ~Symbol() = default;

  virtual const std::string & getName() const { return Name_; }

};

//==============================================================================
// The variable symbol type
//==============================================================================
class VariableSymbol : public Symbol {

  std::shared_ptr<Symbol> Type_;
  bool IsArray_ = false;

public:

  VariableSymbol(const std::string & Name, SourceLocation Loc, 
      std::shared_ptr<Symbol> Type, bool IsArray)
    : Symbol(Name, Loc), Type_(Type), IsArray_(IsArray)
  {}

  virtual ~VariableSymbol() = default;

  const std::shared_ptr<Symbol> getType() const { return Type_; }

};


//==============================================================================
// The function symbol type
//==============================================================================
class FunctionSymbol : public Symbol {

  using symbol_list = std::vector<std::shared_ptr<VariableSymbol>>;

  symbol_list ArgTypes_;
  std::shared_ptr<Symbol> ReturnType_;

public:

  FunctionSymbol(const std::string & Name, SourceLocation Loc,
      const symbol_list & ArgTypes, 
      std::shared_ptr<Symbol> ReturnType = nullptr)
    : Symbol(Name, Loc), ArgTypes_(ArgTypes), ReturnType_(ReturnType)
  {}

  virtual ~FunctionSymbol() = default;

  const std::shared_ptr<Symbol> getReturnType() const { return ReturnType_; }
  const symbol_list & getArgTypes() const { return ArgTypes_; }

};


//==============================================================================
// The symbol table
//==============================================================================
class SymbolTable {

  using table_type = std::map<std::string, Symbol>;
  using iterator = table_type::iterator;

  table_type Symbols_;

public:

  // find a symbol
  std::pair<iterator, bool> find(const std::string name)
  {
    auto it = Symbols_.find(name);
    auto found = (it != Symbols_.end());
      return std::make_pair(it, found);
  }

  // get the size
  std::size_t size() const
  { return Symbols_.size(); }

  // clear the table
  void clear()
  { Symbols_.clear(); }

  // erase elements
  void erase( const std::string & Name )
  { Symbols_.erase(Name); }
  
  // add a symbol
  template<typename...Args>
  std::pair<iterator, bool>
  addSymbol(const std::string & name, Args&&...args)
  {
    return Symbols_.emplace(name, Symbol(std::forward<Args>(args)...));
  }


};

} // namespace

#endif // CONTRA_SYMBOLS_HPP
