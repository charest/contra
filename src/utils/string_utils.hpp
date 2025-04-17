#ifndef CONTRA_STRING_UTILS_HPP
#define CONTRA_STRING_UTILS_HPP

#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

namespace utils {

//! \brief Get a file name.
//! \param [in] str  the input string
//! \return the base of the file name
std::string basename(const std::string & str);

//! \brief Get the extension of a file name.
//! \param [in] str  the input string
//! \return the extension
std::string file_extension(const std::string & str); 

//! \brief Remove the extension from a filename
//! \param [in] str  the input string
//! \return the name without extension
std::string remove_extension(const std::string & str);

//! Split a string s by a delimeter
std::vector<std::string> split(
    const std::string& s,
    char delimiter,
    bool skip_empty = true);

//! remove the extra \ in quoted strings
std::string unescape(const std::string& s);

//! add the extra \ in quoted strings
std::string escape(const std::string& s);

//! sanitize for html 
std::string html(const std::string& s);

//! to lower case
std::string tolower(const std::string& s);
//! to upper case
std::string toupper(const std::string& s);

//! count digits
size_t count_digits(int i);

template<typename T = std::string>
void printLeft(std::ostream & os, int width, char sep, const T & val = std::string())
{
  os << std::left << std::setw(width) << std::setfill(sep) << val;
}

template<typename T = std::string>
void printRight(std::ostream & os, int width, char sep, const T & val = std::string())
{
  os << std::right << std::setw(width) << std::setfill(sep) << val;
}

////////////////////////////////////////////////////////////////////////////////
//! \brief Convert a value to a string.
//! \param [in] x The value to convert to a string.
//! \return the new string
////////////////////////////////////////////////////////////////////////////////
template < typename T >
auto to_string(const T & x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

} // namespace

#endif // CONTRA_STRING_UTILS_HPP
