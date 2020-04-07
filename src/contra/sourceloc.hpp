#ifndef CONTRA_SOURCELOC_HPP
#define CONTRA_SOURCELOC_HPP

#include <iostream>

namespace contra {

class SourceLocation {
  int Line_ = 1;
  int Col_ = 0;
public:
  SourceLocation() = default;
  SourceLocation(int Line, int Col) : Line_(Line), Col_(Col) {}
  int getLine() const { return Line_; }
  int getCol() const { return Col_; }
  int incrementLine() { Line_++; return Line_; }
  int incrementCol() { Col_++; return Col_; }
  void reset() { Line_ = 1; Col_ = 0; }
  void newLine() { Line_++; Col_ = 0; }
  friend std::ostream &operator<<( std::ostream &out, const SourceLocation &obj ) { 
     out << "Line : " << obj.getLine() << ", Col : " << obj.getCol();
     return out;            
  }

};

}

#endif // CONTRA_SOURCELOC_HPP
