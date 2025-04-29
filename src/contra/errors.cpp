#include "stream.hpp"

#include <iostream>
#include <string>

namespace contra {

using pos_type = std::ios::pos_type;
  
std::string get_line(std::istream & in, pos_type lineStart)
{
  in.seekg(lineStart);
  std::string line;
  std::getline(in, line);
  return line;
}

//==============================================================================
/// Count the lines in the file
//==============================================================================
std::pair<size_t,pos_type> count_lines(std::istream & in, pos_type pos)
{
  // seek to beginning of the file
  in.seekg(0);
  size_t lineCount=0;
  pos_type lineStart = 0;

  // Count lines up to 
  char ch;
  while (in.get(ch)) {
    if (ch == '\n') {
      lineCount++;
      lineStart = in.tellg();
    }
    if (in.tellg() >= pos) break;
  }

  return {lineCount, lineStart};
}

//==============================================================================
/// dump out the current line
//==============================================================================
int error(stream_t & is, const std::string & msg)
{
  auto & in = is.in;

  // get current position
  auto currentPos = in.tellg();

  // check if we reached the end of file
  auto is_eof = in.eof();
  in.clear(); // in case we reached end of stream

  if (is_eof) {
    in.seekg(0, std::ios::end);
    currentPos = in.tellg();
  }

  // seek to beginning of the file
  auto [lineCount, lineStart] = count_lines(in, currentPos);

  // get the line with the error
  auto line = get_line(in, lineStart);
  
  // output
  if (is.name.size()) std::cerr << is.name << ":";
  auto col = currentPos - lineStart + is_eof;
  std::cerr << lineCount+1 << ":" << col << ": error: " << msg << std::endl;
  std::cerr << line << std::endl;
  std::cerr << std::string(col-1, ' ') << "^" << std::endl;

  // go back to the original position
  in.seekg(currentPos);
  // seek to end if it was originally at the end
  if (is_eof) in.get();

  return 1;
}

//==============================================================================
/// dump out the current line
//==============================================================================
int error(
  stream_t & is,
  const std::string & msg,
  stream_pos_t pos)
{
  auto & in = is.in;
  auto [beg, end] = pos;

  // get current position
  auto currentPos = in.tellg();

  // check if we reached the end of file
  auto at_eof = in.eof();
  in.clear(); // in case we reached end of stream

  if (beg == EOF) {
    in.seekg(0, std::ios::end);
    beg = in.tellg();
    end = EOF;
  }

  // count lines
  auto [lineCount, lineStart] = count_lines(in, beg);

  // get the line with the error
  auto line = get_line(in, lineStart);
  auto lineLength = line.size();
  
  // output
  if (is.name.size()) std::cerr << is.name << ":";
  auto beg_col = beg - lineStart;
  auto col1 = end - lineStart;
  auto col2 = static_cast<size_t>(beg) + lineLength;
  auto end_col = (col1 > col2) ? col2 : col1;
  auto ncol = end_col > beg_col ? end_col - beg_col : 1;

  std::cerr << lineCount+1 << ":" << beg_col << ": error: " << msg << std::endl;
  std::cerr << line << std::endl;
  std::cerr << std::string(beg_col-1, ' ') << std::string(ncol, '^') << std::endl;
  
  // if we are at the end, of the file,
  // seek to end if it was originally at the end
  if (at_eof) {
    in.seekg(0, std::ios::end);
    in.get();
  }
  // else, go back to the original position
  else {
    in.seekg(currentPos);
  }

  return 1;
}

} // namespace
