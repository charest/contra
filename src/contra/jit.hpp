#ifndef CONTRA_JIT_H
#define CONTRA_JIT_H

#include "config.hpp"
#include "errors.hpp"

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
#include "llvm/ExecutionEngine/OrcMCJITReplacement.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace contra {

class JIT {
public:
  
  using ObjLayerT = llvm::orc::LegacyRTDyldObjectLinkingLayer;

  using Compiler = llvm::orc::SimpleCompiler;
  using CompileLayerT = llvm::orc::LegacyIRCompileLayer<ObjLayerT, Compiler>;
  
  using JITSymbol = llvm::JITSymbol;
  using VModuleKey = llvm::orc::VModuleKey;

  JIT(llvm::TargetMachine* TM) 
    : 
      MM_(std::make_shared<llvm::SectionMemoryManager>()),
      Resolver_(
        llvm::orc::createLegacyLookupResolver(
          ES_,
          [this](const std::string &Name) { return findMangledSymbol(Name); },
          [](llvm::Error Err) { llvm::cantFail(std::move(Err), "lookupFlags failed"); }
        )
      ),
      TM_(TM),
      DL_(TM_->createDataLayout()),
      ObjectLayer_(
        llvm::AcknowledgeORCv1Deprecation,
        ES_,
        [this](VModuleKey) {
          return ObjLayerT::Resources{MM_, Resolver_};
        }
      ),
      CompileLayer_(
        llvm::AcknowledgeORCv1Deprecation,
          ObjectLayer_,
          Compiler(*TM_)
      )
  {
    //EE = std::make_unique<LLVMLinkInOrcMCJITReplacement>(MM, Resolver, TM_);
    std::string ErrMsgStr;
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr); 
#ifdef HAVE_LEGION
    if( llvm::sys::DynamicLibrary::LoadLibraryPermanently(REALM_LIBRARY, &ErrMsgStr) )
      THROW_CONTRA_ERROR(ErrMsgStr);
    if( llvm::sys::DynamicLibrary::LoadLibraryPermanently(LEGION_LIBRARY, &ErrMsgStr) )
      THROW_CONTRA_ERROR(ErrMsgStr);
#endif
  }

  JIT() : JIT(llvm::EngineBuilder().selectTarget()) {}

  auto getTargetMachine() { return TM_.get(); }

  auto addModule(std::unique_ptr<llvm::Module> M) {
    auto K = ES_.allocateVModule();
    llvm::cantFail(CompileLayer_.addModule(K, std::move(M)));
    ModuleKeys_.push_back(K);
    return K;
  }

  void removeModule(VModuleKey K) {
    ModuleKeys_.erase(llvm::find(ModuleKeys_, K));
    llvm::cantFail(CompileLayer_.removeModule(K));
  }

  auto findSymbol(const std::string Name) {
    return findMangledSymbol(mangle(Name));
  }

  void addDeviceJIT(JIT * DevJIT) { DeviceJIT_ = DevJIT; }

private:

  std::string mangle(const std::string &Name);
  JITSymbol findMangledSymbol(const std::string &Name);

  llvm::orc::ExecutionSession ES_;
  std::shared_ptr<llvm::SectionMemoryManager> MM_;
  std::shared_ptr<llvm::orc::SymbolResolver> Resolver_;
  std::unique_ptr<llvm::TargetMachine> TM_;
  const llvm::DataLayout DL_;
  ObjLayerT ObjectLayer_;
  CompileLayerT CompileLayer_;
  std::vector<VModuleKey> ModuleKeys_;
  //std::unique_ptr<llvm::ExecutionEngine> EE;
  
  JIT* DeviceJIT_ = nullptr;
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
