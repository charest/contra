#include "jit.hpp"

using namespace llvm;

namespace contra {
  
//==============================================================================
std::string JIT::mangle(StringRef Name) {
  std::string MangledName;
  {
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL_);
  }
  return MangledName;
}

//==============================================================================
JIT::JITSymbol JIT::findMangledSymbol(StringRef Name) {
  
  using JITSymbolFlags = JITSymbolFlags;
  using RTDyldMemoryManager = RTDyldMemoryManager;

#ifdef _WIN32
  // The symbol lookup of ObjectLinkingLayer uses the SymbolRef::SF_Exported
  // flag to decide whether a symbol will be visible or not, when we call
  // IRCompileLayer::findSymbolIn with ExportedSymbolsOnly set to true.
  //
  // But for Windows COFF objects, this flag is currently never set.
  // For a potential solution see: https://reviews.llvm.org/rL258665
  // For now, we allow non-exported symbols on Windows as a workaround.
  const bool ExportedSymbolsOnly = false;
#else
  const bool ExportedSymbolsOnly = true;
#endif

  // Search modules in reverse order: from last added to first added.
  // This is the opposite of the usual search order for dlsym, but makes more
  // sense in a REPL where we want to bind to the newest available definition.
  for (auto H : make_range(ModuleKeys_.rbegin(), ModuleKeys_.rend()))
    if (auto Sym = CompileLayer_.findSymbolIn(H, Name.str(), ExportedSymbolsOnly))
      return Sym;

  // If we can't find the symbol in the JIT, try looking in the host process.
  if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name.str()))
    return JITSymbol(SymAddr, JITSymbolFlags::Exported);

#ifdef _WIN32
  // For Windows retry without "_" at beginning, as RTDyldMemoryManager uses
  // GetProcAddress and standard libraries like msvcrt.dll use names
  // with and without "_" (for example "_itoa" but "sin").
  if (Name.length() > 2 && Name[0] == '_')
    if (auto SymAddr =
            RTDyldMemoryManager::getSymbolAddressInProcess(Name.substr(1)))
      return JITSymbol(SymAddr, JITSymbolFlags::Exported);
#endif

  return nullptr;
}

} // namespace
