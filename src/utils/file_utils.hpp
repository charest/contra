#ifndef CONTRA_FILE_UTILS_HPP
#define CONTRA_FILE_UTILS_HPP

#include <fstream>

namespace utils {

inline bool file_exists (const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

} // namespace

#endif // CONTRA_FILE_UTILS_HPP
