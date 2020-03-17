#ifndef CONTRA_ERRORS_HPP
#define CONTRA_ERRORS_HPP

#include "formatter.hpp"
#include "sourceloc.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace contra {

class ContraError;
class CodeError;

//==============================================================================
// Abstrct visitor for errors
//==============================================================================
class ErrorDispatcher {
public:
  virtual ~ErrorDispatcher() = default;
  virtual void dispatch(const ContraError&) const = 0;
  virtual void dispatch(const CodeError&) const = 0;
};

//==============================================================================
// General contra error
//==============================================================================
class ContraError : public std::runtime_error
{
public:
  ContraError() : std::runtime_error( "general contra error" ) {} 
  ContraError(const char *str) : std::runtime_error(str) {}
  ContraError(const std::string & str) : std::runtime_error(str) {}
  virtual ~ContraError() {}
  virtual void accept(const ErrorDispatcher& dispatcher) const 
  { dispatcher.dispatch(*this); }
};

//==============================================================================
// Base code error
//==============================================================================
class CodeError : public ContraError
{
  SourceLocation Loc_;
public:
  CodeError(const char *str, SourceLocation Loc) : ContraError(str), Loc_(Loc) {}
  CodeError(const std::string & str, SourceLocation Loc) : ContraError(str), Loc_(Loc) {}
  SourceLocation getLoc() const { return Loc_; }
  virtual ~CodeError() {}
  virtual void accept(const ErrorDispatcher& dispatcher) const override
  { dispatcher.dispatch(*this); }
};


//==============================================================================
// Invalid identifier accessed
//==============================================================================
class NameError : public CodeError
{
public:
  NameError(const char *str, SourceLocation Loc) : CodeError(str, Loc) {}
  NameError(const std::string & str, SourceLocation Loc) : CodeError(str, Loc) {}
  virtual ~NameError() {}
};

//==============================================================================
// A syntax error
//==============================================================================
class SyntaxError : public CodeError
{
public:
  SyntaxError(const char *str, SourceLocation Loc) : CodeError(str, Loc) {}
  SyntaxError(const std::string & str, SourceLocation Loc) : CodeError(str, Loc) {}
  virtual ~SyntaxError() {}
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
