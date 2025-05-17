#ifndef CONTRA_STREAM_HPP
#define CONTRA_STREAM_HPP

#include <istream>
#include <string>
#include <vector>

namespace contra {

//==============================================================================
/// Stream position
//==============================================================================
struct stream_pos_t {
  size_t begin, end;
  auto length() const { return end - begin; }
};


struct stream_t {

  std::string buffer;
  std::string name;
  std::vector<size_t> newlines;

  std::string at(stream_pos_t pos) const
  { return buffer.substr( pos.begin, pos.length() ); }

};

stream_t make_stream(std::istream & in, const std::string & name = "");

} // namespace

#endif
