#include "dllexport.h"
#include "dopevector.hpp"
#include "librt.hpp"
#include "math.hpp"
#include "print.hpp"

namespace librt {

//==============================================================================
// install the library functions available by default
//==============================================================================
std::map<std::string, RunTimeLib::InstallFunctionPointer>
  RunTimeLib::InstallMap = {
    {"print",installPrint},
    {"allocate",installAllocate},
    {"deallocate",installDeAllocate},
    {"myabs",installAbs},
    {"mysqrt",installSqrt},
    {"fabs",installCAbs},
    {"fmax",installCMax},
    {"sqrt",installCSqrt},
  };

}
