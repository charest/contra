#ifndef CONTRA_SOURCELOC_HPP
#define CONTRA_SOURCELOC_HPP

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
};

}

#endif // CONTRA_SOURCELOC_HPP
