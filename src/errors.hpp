#ifndef CONTRA_ERRORS_HPP
#define CONTRA_ERRORS_HPP

#include <iostream>

namespace contra {

/// LogError* - These are little helper functions for error handling.
inline void LogError(const char *Str) {
  std::cerr << "Error: " << Str << "\n";
}

} // namespace 

#endif // CONTRA_ERRORS_HPP
