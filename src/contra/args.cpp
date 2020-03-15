#include "args.hpp"
#include "string_utils.hpp"

#include <getopt.h>
#include <iostream>
#include <vector>

namespace contra {

///////////////////////////////////////////////////////////////////////////////
//! \brief Parse the argument list
///////////////////////////////////////////////////////////////////////////////
auto parseArguments(
  int argc,
  char** argv,
  const option * long_options,
  const char * short_options,
  std::map<std::string, std::string> & key_value_pair
) {

  // reset getopts global variable
  optind = 0;

  // getopt_long stores the option index here.
  int option_index = 0;

  // if no arguments, set c to -1 to skip while lop
  int c = ( argc > 1 ) ? 0 : -1;

  // make a copy to avoid getopt permuting invalid options
  std::vector<char*> argvcopy(argv, argv + argc);

  while (c != -1) {
    c = getopt_long(argc, argvcopy.data(), short_options, long_options, &option_index);
    auto c_char = static_cast<char>(c);
    auto c_str = to_string( c_char );
    // finished with arguments
    if ( c == -1 )
      break;
    // long options that set a flag
    else if (c == 0)
      key_value_pair[long_options[option_index].name] =
        optarg ? optarg : "";
    // standard short/long option
    else
      key_value_pair[c_str] = optarg ? optarg : "";
    
  }
    
  // rest of arguments are positional
  if (optind < argc && argc > 1) {
    auto & positional = key_value_pair["__positional"];
    while (optind < argc)
      positional += argv[optind++] + std::string(";");
    if (positional.back() == ';') positional.pop_back();
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Process the argument list for this app.
///////////////////////////////////////////////////////////////////////////////
int processArguments(
    int argc,
    char** argv,
    std::map<std::string, std::string> & args) 
{

  // the usage stagement
  auto print_usage = [&argv]() {
    std::cout << "Usage: " << argv[0]
              << " [--compile,-c]"
              << " [--debug,-g]"
              << " [--dump-ir,-i IR_FILE]"
              << " [--dump-dot,-d DOT_FILE]"
              << " [--help,-h]"
              << " [--optimize,-O]"
              << " [--output,-o OUTPUT_FILE]"
              << " [--verbose,-v]"
              << " [SOURCE_FILE]"
              << std::endl << std::endl;
    std::cout << "\t--compile:\t Compile provided SOURCE_FILE." << std::endl;
    std::cout << "\t--debug:\t Turn off optimizations." << std::endl;
    std::cout << "\t--dump-ir:\t Dump IR." << std::endl;
    std::cout << "\t--dump-dot:\t Dump AST." << std::endl;
    std::cout << "\t--optimize:\t Optimize." << std::endl;
    std::cout << "\t--output:\t Output object file to OUTPUT_FILE." << std::endl;
    std::cout << "\t--verbose:\t Print debug information." << std::endl;
    std::cout << "\t--help:\t\t Print a help message." << std::endl;
  };

  // Define the options
  struct option long_options[] =
    {
      {"debug",           no_argument, 0, 'g'},
      {"compile",         no_argument, 0, 'c'},
      {"help",            no_argument, 0, 'h'},
      {"dump-ir",   required_argument, 0, 'i'},
      {"dump-dot",  required_argument, 0, 'd'},
      {"output",    required_argument, 0, 'o'},
      {"optimize",  required_argument, 0, 'O'},
      {"verbose",         no_argument, 0, 'v'},
      {0, 0, 0, 0}
    };
  const char * short_options = "d:ghi:cOo:v";

  // parse the arguments
  auto ret = parseArguments(argc, argv, long_options, short_options, args);

  // process the simple ones
  if ( args.count("h") ) print_usage();

  return ret;
}

} // namespace
