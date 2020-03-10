#ifndef CONTRA_ERRORS_HPP
#define CONTRA_ERRORS_HPP

#include "formatter.hpp"
#include "sourceloc.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace contra {

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
// Invalid identifier accessed
//==============================================================================
class NameError : public ContraError
{
  SourceLocation Loc_;
public:
  NameError(const char *str, SourceLocation Loc) : ContraError(str), Loc_(Loc) {}
  NameError(const std::string & str, SourceLocation Loc) : ContraError(str), Loc_(Loc) {}
};

//==============================================================================
// A syntax error
//==============================================================================
class SyntaxError : public ContraError
{
  SourceLocation Loc_;
public:
  SyntaxError(const char *str, SourceLocation Loc) : ContraError(str), Loc_(Loc) {}
  SyntaxError(const std::string & str, SourceLocation Loc) : ContraError(str), Loc_(Loc) {}
};


////////////////////////////////////////////////////////////////////////////////
//! \brief Raise a syntax error.
////////////////////////////////////////////////////////////////////////////////
#define THROW_SYNTAX_ERROR(msg,loc)                                            \
  do {                                                                         \
    throw ::contra::SyntaxError(Formatter() << msg, loc );                     \
  } while(0)

////////////////////////////////////////////////////////////////////////////////
//! \brief Raise a name error.
////////////////////////////////////////////////////////////////////////////////
#define THROW_NAME_ERROR(msg,loc)                                              \
  do {                                                                         \
    throw ::contra::SyntaxError(Formatter() << msg, loc);                      \
  } while(0)


////////////////////////////////////////////////////////////////////////////////
//! \brief Raise a general runtime error.
////////////////////////////////////////////////////////////////////////////////
#define THROW_CONTRA_ERROR(msg)                                                \
  do {                                                                         \
    std::cerr << "General runtime error:" << std::endl;                        \
    throw ::contra::ContraError(Formatter() << msg  );                         \
  } while(0)

////////////////////////////////////////////////////////////////////////////////
//! \brief Raise a general runtime error.
////////////////////////////////////////////////////////////////////////////////
#define THROW_IMPLEMENTED_ERROR(msg)                                           \
  do {                                                                         \
    std::cerr << "Implementation error:" << std::endl;                         \
    throw ::contra::ContraError(Formatter() << msg  );                         \
  } while(0)



} // namespace 

#endif // CONTRA_ERRORS_HPP
