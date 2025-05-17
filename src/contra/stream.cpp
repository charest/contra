#include "stream.hpp"
#include "utils/string_utils.hpp"

namespace contra {

stream_t make_stream(std::istream & in, const std::string & name)
{
  stream_t strm;
  strm.name = name;

  in.seekg(0, std::ios::end);
  auto size = in.tellg();
  in.seekg(0, std::ios::beg);

  strm.buffer.resize(size);
  if (in.read(strm.buffer.data(), size))
    strm.newlines = utils::newline_positions(strm.buffer);

  return strm;
}

} // namespace
