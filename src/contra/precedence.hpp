#ifndef CONTRA_PRECEDENCE_HPP
#define CONTRA_PRECEDENCE_HPP

#include "token.hpp"

#include <map>

namespace contra {

struct BinopPrecedenceResult {
  bool found = false;
  int precedence = -1;
};


class BinopPrecedence {

  std::map<int, int> Precedence_;

public:

  void add(int key, int val)
  { Precedence_[key] = val; }

  BinopPrecedenceResult find( char key ) const
  {
    auto it = Precedence_.find(key);
    if ( it != Precedence_.end() )
      return {true, it->second};
    else
      return {false, -1};
  }
  
  int find_v2( int key ) const
  {
    auto it = Precedence_.find(key);
    if ( it != Precedence_.end() )
      return it->second;
    return -1;
  }


  auto count(int key) const
  { return Precedence_.count(key); }

  int operator[]( int key ) const { return Precedence_.at(key); }
  int& operator[]( int key ) { return Precedence_[key]; }
  int at( int key ) const { return Precedence_.at(key); }
  int& at( int key ) { return Precedence_.at(key); }

  
};

/// Main precedence builder
BinopPrecedence make_contra_precedence();


} // namespace

#endif // CONTRA_PRECEDENCE_HPP
