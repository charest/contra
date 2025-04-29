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
  
  std::map<int, int> binary_left;
  std::map<int, int> binary_right;
  std::map<int, int> unary;


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
  
  auto count(int key) const
  { return Precedence_.count(key); }

  int operator[]( int key ) const { return Precedence_.at(key); }
  int& operator[]( int key ) { return Precedence_[key]; }
  int at( int key ) const { return Precedence_.at(key); }
  int& at( int key ) { return Precedence_.at(key); }

  int findLeft(int tok) const
  {
    auto it = binary_left.find(tok);
    if (it != binary_left.end()) return it->second;
    else return -1;
  }
  int findRight(int tok) const
  {
    auto it = binary_right.find(tok);
    if (it != binary_right.end()) return it->second;
    else return -1;
  }
  
  int findBinary(int tok) const
  {
    auto res = findLeft(tok);
    if (res!=-1) return res;
    res = findRight(tok);
    if (res!=-1) return res;
    return -1;
  }

  int findUnary(int tok) const
  {
    auto it = unary.find(tok);
    if (it != unary.end()) return it->second;
    else return -1;
  }
  
};

} // namespace

#endif // CONTRA_PRECEDENCE_HPP
