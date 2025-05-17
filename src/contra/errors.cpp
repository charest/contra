#include "stream.hpp"
#include "utils/string_utils.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace contra {

//==============================================================================
/// Count the lines in the file
//==============================================================================
std::pair<size_t,size_t> count_lines(const std::vector<size_t> & lines, size_t pos)
{
  if (lines.empty() || pos < lines[0])
    return {0, 0};

  auto it = std::upper_bound(lines.begin(), lines.end(), pos);
  if (it != lines.end())
    return {std::distance(lines.begin(), it), *std::prev(it)+1};

  return {lines.size(), lines.back()+1};
}

//==============================================================================
/// dump out the current line
//==============================================================================
int error(const stream_t & is, const std::string & msg, size_t pos)
{
  // seek to beginning of the file
  auto [lineCount, lineStart] = count_lines(is.newlines, pos);

  // get the line with the error
  auto line = utils::extract_to_newline(is.buffer, lineStart);
  
  // output
  if (is.name.size()) std::cerr << is.name << ":";
  auto col = pos - lineStart + 1;
  std::cerr << lineCount+1 << ":" << col << ": error: " << msg << std::endl;
  std::cerr << line << std::endl;
  std::cerr << std::string(col-1, ' ') << "^" << std::endl;

  return 1;
}

//==============================================================================
/// dump out the current line
//==============================================================================
int error(const stream_t & is, const std::string & msg, const stream_pos_t pos)
{
  // figure out the line start
  auto [lineNo, lineStart] = count_lines(is.newlines, pos.begin);

  // get the line
  auto line = utils::extract_to_newline(is.buffer, lineStart);
  auto lineLen = line.size();

  // output
  if (is.name.size()) std::cerr << is.name << ":";
  auto colNo = pos.begin - lineStart;
  auto width = pos.end - pos.begin;

  std::cerr << lineNo+1 << ":" << colNo+1 << ": error: " << msg << std::endl;
  std::cerr << line << std::endl;
  std::cerr << std::string(colNo, ' ') << std::string(width, '^') << std::endl;

  return 1;
}

} // namespace
