#ifndef CONTRA_JIT_H
#define CONTRA_JIT_H

#include "config.hpp"
#include "errors.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace contra {

static llvm::ExitOnError ExitOnErr;

class JIT {
public:
  
  using JITSymbol = llvm::JITSymbol;
  using Resource = llvm::orc::ResourceTrackerSP;
  
  JIT( std::unique_ptr<llvm::orc::ExecutionSession> ES,
       llvm::orc::JITTargetMachineBuilder JTMB,
       llvm::DataLayout DL)
    :
      ES_(std::move(ES)),
      DL_(std::move(DL)),
      Mangle_(*ES_, DL_),
      ObjectLayer_(*ES_, []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      CompileLayer_(*ES_, ObjectLayer_,
                     std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(JTMB))),
      MainJD_(ES_->createBareJITDylib("<main>")),
      Context_(std::make_unique<llvm::LLVMContext>())
  {
    MainJD_.addGenerator(
        llvm::cantFail(
          llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL_.getGlobalPrefix())));
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      ObjectLayer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
      ObjectLayer_.setAutoClaimResponsibilityForObjectSymbols(true);
    }

    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
#ifdef HAVE_LEGION
    std::string ErrMsgStr;
    if( llvm::sys::DynamicLibrary::LoadLibraryPermanently(REALM_LIBRARY, &ErrMsgStr) )
      THROW_CONTRA_ERROR(ErrMsgStr);
    if( llvm::sys::DynamicLibrary::LoadLibraryPermanently(LEGION_LIBRARY, &ErrMsgStr) )
      THROW_CONTRA_ERROR(ErrMsgStr);
#endif
  }

  static llvm::Expected<std::unique_ptr<JIT>> Create() {
    auto EPC = llvm::orc::SelfExecutorProcessControl::Create();
    if (!EPC)
      return EPC.takeError();

    auto ES = std::make_unique<llvm::orc::ExecutionSession>(std::move(*EPC));

    llvm::orc::JITTargetMachineBuilder JTMB(
        ES->getExecutorProcessControl().getTargetTriple());

    auto DL = JTMB.getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return std::make_unique<JIT>(std::move(ES), std::move(JTMB), std::move(*DL));
  }

  const auto & getDataLayout() { return DL_; }

  llvm::orc::ResourceTrackerSP addModule(
    llvm::orc::ThreadSafeModule TSM,
    llvm::orc::ResourceTrackerSP RT = nullptr
  )
  {
    if (!RT) RT = MainJD_.getDefaultResourceTracker();
    ExitOnErr(CompileLayer_.add(RT, std::move(TSM)));
    return RT;
  }

  llvm::orc::ResourceTrackerSP addModule(
    std::unique_ptr<llvm::Module> M, 
    llvm::orc::ResourceTrackerSP RT = nullptr
  )
  {
    auto TSM = llvm::orc::ThreadSafeModule( std::move(M), Context_ );
    return addModule(std::move(TSM), RT);
  }

  llvm::orc::ResourceTrackerSP createResource() const
  { return MainJD_.createResourceTracker(); }

  void removeModule(llvm::orc::ResourceTrackerSP RT)
  {
    ExitOnErr(RT->remove());
  }

  llvm::Expected<llvm::orc::ExecutorSymbolDef> findSymbol(llvm::StringRef Name)
  {
    return ES_->lookup({&MainJD_}, Mangle_(Name.str()));
  }

  auto getContext() { return Context_.getContext(); }

private:

  std::string mangle(llvm::StringRef Name);
  JITSymbol findMangledSymbol(llvm::StringRef Name);

  std::unique_ptr<llvm::orc::ExecutionSession> ES_;
  //llvm::orc::ExecutionSession ES_;
  //std::shared_ptr<llvm::orc::SymbolResolver> Resolver_;
  //std::unique_ptr<llvm::TargetMachine> TM_;
  const llvm::DataLayout DL_;
  llvm::orc::MangleAndInterner Mangle_;
  llvm::orc::RTDyldObjectLinkingLayer ObjectLayer_;
  llvm::orc::IRCompileLayer CompileLayer_;
  //std::unique_ptr<llvm::ExecutionEngine> EE;
  
  llvm::orc::JITDylib &MainJD_;
  llvm::orc::ThreadSafeContext Context_;
  
};

} // end namespace

#endif // CONTRA_CONTRAJIT_H
