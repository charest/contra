#ifndef CONTRA_SOURCELOC_HPP
#define CONTRA_SOURCELOC_HPP

#include <iostream>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Base source location
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
/// Identifier start/stop
////////////////////////////////////////////////////////////////////////////////
class LocationRange {
  SourceLocation Begin_;
  SourceLocation End_;
public:
  LocationRange() = default;
  LocationRange(const SourceLocation & Begin, const SourceLocation & End)
    : Begin_(Begin), End_(End) {}
  const auto & getBegin() const { return Begin_; }
  const auto & getEnd() const { return End_; }
  friend std::ostream &operator<<( std::ostream &out, const LocationRange &obj ) { 
     out << "Begin :: " << obj.Begin_ << std::endl;
     out << "End   :: " << obj.End_ << std::endl;
     return out;            
  }
};

}

#endif // CONTRA_SOURCELOC_HPP
