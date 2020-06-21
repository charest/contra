#include "kokkos.hpp"

#include "utils/llvm_utils.hpp"
  
////////////////////////////////////////////////////////////////////////////////
// Legion tasker
////////////////////////////////////////////////////////////////////////////////

namespace contra {

using namespace llvm;
using namespace utils;

//==============================================================================
// Constructor
//==============================================================================
KokkosTasker::KokkosTasker(utils::BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{}

//==============================================================================
// start runtime
//==============================================================================
Value* KokkosTasker::startRuntime(Module &TheModule, int Argc, char ** Argv)
{

  auto ArgcV = llvmValue(TheContext_, Int32Type_, Argc);

  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argc; ++i)
    ArgVs.emplace_back( llvmString(TheContext_, TheModule, Argv[i]) );

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  auto ArgvV = llvmArray(TheContext_, TheModule, ArgVs, {ZeroC, ZeroC});

  std::vector<Value*> StartArgVs = { ArgcV, ArgvV };
  auto RetI = TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_runtime_start",
      Int32Type_,
      StartArgVs,
      "start");
  return RetI;
}

//==============================================================================
// stop runtime
//=============================================================================
void KokkosTasker::stopRuntime(Module &TheModule)
{
  TheHelper_.callFunction(
      TheModule,
      "contra_kokkos_runtime_stop",
      VoidType_,
      {});
}


}
