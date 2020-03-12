#ifndef CONTRA_IDENTIFIER_HPP
#define CONTRA_IDENTIFIER_HPP

#include "sourceloc.hpp"
#include <string>

namespace contra {

class Identifier {
  std::string Name_;
  SourceLocation Loc_;
  
public:

  Identifier() = default;

  Identifier(const std::string N, SourceLocation L) : Name_(N), Loc_(L) 
  {}

  const std::string & getName() const { return Name_; }
  SourceLocation getLoc() const { return Loc_; }
};

}

#endif // CONTRA_IDENTIFIER_HPP
