#ifndef CONTRA_ERRORS_HPP
#define CONTRA_ERRORS_HPP

#include "formatter.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace contra {

/// LogError* - These are little helper functions for error handling.
inline void LogError(const char *Str) {
  std::cerr << "Error: " << Str << "\n";
}

//==============================================================================
// General base contra error
//==============================================================================
class ContraError : public std::runtime_error
{
public:
  ContraError() : std::runtime_error( "general contra error" ) {}
  
  ContraError(const char *str) : std::runtime_error(str) {}

  ContraError(const std::string & str) : std::runtime_error(str) {}
};


//==============================================================================
//==============================================================================
class NameError : public ContraError
{
public:
  NameError(const char *str) : ContraError(str) {}
  NameError(const std::string & str) : ContraError(str) {}
};

//==============================================================================
//==============================================================================
class SyntaxError : public ContraError
{
public:
  SyntaxError(const char *str) : ContraError(str) {}

  SyntaxError(const std::string & str) : ContraError(str) {}
};


////////////////////////////////////////////////////////////////////////////////
//! \brief Raise a runtime error.
////////////////////////////////////////////////////////////////////////////////
#define THROW_SYNTAX_ERROR(msg,line)                                           \
  do {                                                                         \
    std::cerr << "Syntax error on line " << line << std::endl;                 \
    throw ::contra::SyntaxError(Formatter() << msg );                          \
  } while(0)

////////////////////////////////////////////////////////////////////////////////
//! \brief Raise a runtime error.
////////////////////////////////////////////////////////////////////////////////
#define THROW_NAME_ERROR(msg,line)                                             \
  do {                                                                         \
    std::cerr << "Name error on line " << line << std::endl;                   \
    throw ::contra::NameError(Formatter() << "Unknown specifier '" <<  msg     \
        << "'" );                                                              \
  } while(0)



} // namespace 

#endif // CONTRA_ERRORS_HPP
