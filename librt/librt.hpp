#ifndef CONTRA_RTLIB_HPP
#define CONTRA_RTLIB_HPP

#include <map>
#include <string>

namespace llvm {
class Function;
class LLVMContext;
class Module;
}

namespace librt {

//==============================================================================
// Class to keep track of available runtime functions
//==============================================================================
struct RunTimeLib {

  typedef llvm::Function* (*InstallFunctionPointer) (llvm::LLVMContext &, llvm::Module &);
  static std::map<std::string, InstallFunctionPointer> InstallMap;

  static llvm::Function* tryInstall(llvm::LLVMContext &TheContext,
      llvm::Module &TheModule, const std::string & Name)
  {
    auto it = InstallMap.find(Name);
    if (it != InstallMap.end())
      return it->second(TheContext, TheModule);
    return nullptr;
  }

};

}


#endif //CONTRA_RTLIB_HPP
