#ifndef CONTRA_SYMBOLS_HPP
#define CONTRA_SYMBOLS_HPP

#include "sourceloc.hpp"
#include "vartype.hpp"

#include <map>
#include <string>

namespace contra {

//==============================================================================
// The symbol data type
//==============================================================================
class Symbol {

  VarTypes Type_ = VarTypes::Void;
  SourceLocation Loc_;
  bool IsArray_ = false;

public:

  Symbol() = default;

  Symbol(VarTypes Type, SourceLocation Loc, bool IsArray = false) :
    Type_(Type), Loc_(Loc), IsArray_(IsArray)
  {}

  VarTypes getType() const { return Type_; }
};

//==============================================================================
// The symbol table
//==============================================================================
class SymbolTable {

  using table_type = std::map<std::string, Symbol>;

  table_type Symbols_;

public:

  // public types
  using iterator = table_type::iterator;
  using const_iterator = table_type::const_iterator;

  // find a symbol
  iterator find(const std::string name)
  { return Symbols_.find(name); }
  
  const_iterator find(const std::string name) const
  { return Symbols_.find(name); }

  // get the size
  std::size_t size() const
  { return Symbols_.size(); }

  // clear the table
  void clear()
  { Symbols_.clear(); }

  // erase elements
  iterator erase( iterator pos )
  { return Symbols_.erase(pos); }

  iterator erase( const_iterator pos )
  { return Symbols_.erase(pos); }

  // add a symbol
  template<typename...Args>
  std::pair<iterator, bool>
  addSymbol(const std::string & name, Args&&...args)
  { return Symbols_.emplace(name, Symbol(std::forward<Args>(args)...)); }

  void addSymbols( const std::vector< std::pair<std::string, Symbol> > & List )
  {
    for ( const auto & i : List )
      Symbols_.emplace(i.first, i.second);
  }

  // beginning and end iterators
  iterator begin()
  { return Symbols_.begin(); }
  
  const_iterator begin() const
  { return Symbols_.begin(); }

  iterator end()
  { return Symbols_.end(); }
  
  const_iterator end() const
  { return Symbols_.end(); }


};

} // namespace

#endif // CONTRA_SYMBOLS_HPP
