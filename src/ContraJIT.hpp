#ifndef CONTRA_CONTRAJIT_H
#define CONTRA_CONTRAJIT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace contra {

class ContraJIT {

  using JITSymbol = llvm::JITSymbol;
  using VModuleKey = llvm::orc::VModuleKey;

public:

  using ObjLayerT = llvm::orc::LegacyRTDyldObjectLinkingLayer;

  using Compiler = llvm::orc::SimpleCompiler;
  using CompileLayerT = llvm::orc::LegacyIRCompileLayer<ObjLayerT, Compiler>;

  ContraJIT() 
    : Resolver(
        llvm::orc::createLegacyLookupResolver(
          ES,
          [this](const std::string &Name) { return findMangledSymbol(Name); },
          [](llvm::Error Err) { llvm::cantFail(std::move(Err), "lookupFlags failed"); }
        )
      ),
      TM(llvm::EngineBuilder().selectTarget()),
      DL(TM->createDataLayout()),
      ObjectLayer(
        llvm::AcknowledgeORCv1Deprecation,
        ES,
        [this](VModuleKey) {
          return ObjLayerT::Resources{
            std::make_shared<llvm::SectionMemoryManager>(), Resolver
          };
        }
      ),
      CompileLayer(
        llvm::AcknowledgeORCv1Deprecation,
          ObjectLayer,
          Compiler(*TM)
      ) 
  {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  auto &getTargetMachine() { return *TM; }

  auto addModule(std::unique_ptr<llvm::Module> M) {
    auto K = ES.allocateVModule();
    llvm::cantFail(CompileLayer.addModule(K, std::move(M)));
    ModuleKeys.push_back(K);
    return K;
  }

  void removeModule(VModuleKey K) {
    ModuleKeys.erase(llvm::find(ModuleKeys, K));
    llvm::cantFail(CompileLayer.removeModule(K));
  }

  auto findSymbol(const std::string Name) {
    return findMangledSymbol(mangle(Name));
  }

private:

  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      llvm::raw_string_ostream MangledNameStream(MangledName);
      llvm::Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  JITSymbol findMangledSymbol(const std::string &Name) {
  
    using JITSymbolFlags = llvm::JITSymbolFlags;
    using RTDyldMemoryManager = llvm::RTDyldMemoryManager;

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
    for (auto H : llvm::make_range(ModuleKeys.rbegin(), ModuleKeys.rend()))
      if (auto Sym = CompileLayer.findSymbolIn(H, Name, ExportedSymbolsOnly))
        return Sym;

    // If we can't find the symbol in the JIT, try looking in the host process.
    if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
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

  llvm::orc::ExecutionSession ES;
  std::shared_ptr<llvm::orc::SymbolResolver> Resolver;
  std::unique_ptr<llvm::TargetMachine> TM;
  const llvm::DataLayout DL;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  std::vector<VModuleKey> ModuleKeys;
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
