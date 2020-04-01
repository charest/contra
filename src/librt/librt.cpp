#include "dllexport.h"
#include "dopevector.hpp"
#include "librt.hpp"
#include "math.hpp"
#include "print.hpp"

namespace librt {

// static initialization
std::map<std::string, RunTimeLib::SetupFunctionPointer>
  RunTimeLib::SetupMap;

std::map<std::string, RunTimeLib::LlvmFunctionPointer>
  RunTimeLib::InstallMap;

std::map<std::string, RunTimeLib::FunctionPointer>
  RunTimeLib::SemantecMap;

//==============================================================================
// install the library functions available by default
//==============================================================================
void RunTimeLib::setup(llvm::LLVMContext & TheContext)
{
    _setup<Print, Allocate, DeAllocate, CAbs, CMax, CSqrt>();
    for (auto & entry : SetupMap) entry.second(TheContext);
}

}
