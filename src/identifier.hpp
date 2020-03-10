#ifndef CONTRA_IDENTIFIER_HPP
#define CONTRA_IDENTIFIER_HPP

#include "sourceloc.hpp"
#include <string>

namespace contra {

struct Identifier {
  std::string Name;
  SourceLocation Loc;

  Identifier() = default;

  Identifier(const std::string N, SourceLocation L) :
    Name(N), Loc(L) 
  {}
};

}

#endif // CONTRA_IDENTIFIER_HPP
