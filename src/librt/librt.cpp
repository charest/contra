#include "dllexport.h"
#include "dopevector.hpp"
#include "librt.hpp"
#include "math.hpp"
#include "print.hpp"

#include "contra/symbols.hpp"

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
    _setup<Print, DopeVectorAllocate, DopeVectorDeAllocate, DopeVectorCopy,
      CAbs, CMax, CMin, CSqrt>();
    for (auto & entry : SetupMap) entry.second(TheContext);
}

//==============================================================================
// install the library functions available by default
//==============================================================================
std::unique_ptr<contra::FunctionDef>
RunTimeLib::tryInstall(const std::string & Name)
{
  auto it = SemantecMap.find(Name);
  if (it != SemantecMap.end())
    return it->second();
  return std::unique_ptr<contra::BuiltInFunction>(nullptr);
}

}
