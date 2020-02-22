#include "string_utils.hpp"

#include <iostream>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
//! \brief Get a file name.
//! \param [in] str  the input string
//! \return the base of the file name
////////////////////////////////////////////////////////////////////////////////
std::string basename(const std::string & str)
{
#ifdef _WIN32
  char sep = '\\';
#else
  char sep = '/';
#endif

  auto i = str.rfind( sep, str.length() );
  if ( i != std::string::npos )
    return str.substr(i+1, str.length()-1);
  else
    return str;
}

////////////////////////////////////////////////////////////////////////////////
//! \brief Get the extension of a file name.
//! \param [in] str  the input string
//! \return the extension
////////////////////////////////////////////////////////////////////////////////
std::string file_extension(const std::string & str) 
{
  auto base = basename(str);
  auto i = base.rfind( '.', base.length() );

  if ( i != std::string::npos )
    return base.substr(i+1, base.length()-1);
  else
    return "";
 
}

////////////////////////////////////////////////////////////////////////////////
//! \brief Remove the extension from a filename
//! \param [in] str  the input string
//! \return the name without extension
////////////////////////////////////////////////////////////////////////////////
std::string remove_extension(const std::string & str) {
    auto lastdot = str.find_last_of(".");
    if (lastdot == std::string::npos) return str;
    return str.substr(0, lastdot);
}

////////////////////////////////////////////////////////////////////////////////
/// Split a string by a delimeter
////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

////////////////////////////////////////////////////////////////////////////////
// convert remove the extra \ on quoted strings
////////////////////////////////////////////////////////////////////////////////
std::string unescape(const std::string& s)
{
  std::string res;
  std::string::const_iterator it = s.begin();
  while (it != s.end())
  {
    char c = *it++;
    if (c == '\\' && it != s.end())
    {
      switch (*it++) {
      case '\\': c = '\\'; break;
      case 'n': c = '\n'; break;
      case 't': c = '\t'; break;
      // all other escapes
      default: 
        // invalid escape sequence - skip it. alternatively you can copy
        // it as is, throw an exception...
        continue;
      }
    }
    res += c;
  }

  return res;
}

////////////////////////////////////////////////////////////////////////////////
// add the extra \ on quoted strings
////////////////////////////////////////////////////////////////////////////////
std::string escape(const std::string& s)
{
  std::string res;
  std::string::const_iterator it = s.begin();

  while (it != s.end())
  {
    auto c = *it++;
    std::string str(1, c);
    switch (c) {
    case '\n': str = "\\n"; break;
    case '\t': str = "\\t"; break;
    case '\\': str = "\\\\"; break;
    }
    res += str;
  }

  return res;
}

} // namespace
