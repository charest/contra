#include "config.hpp"

#include "hpx.hpp"
#include "hpx_rt.hpp"

#include "llvm/IR/IRBuilder.h"

using namespace llvm;
using namespace utils;
  
namespace contra {

////////////////////////////////////////////////////////////////////////////////
// HPX tasker args
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
// Constructor
//==============================================================================
HpxTasker::HpxTasker(BuilderHelper & TheHelper)
  : AbstractTasker(TheHelper)
{}

//==============================================================================
// Destructor
//==============================================================================
HpxTasker::~HpxTasker()
{
  if (isStarted()) contra_hpx_shutdown(); 
}

//==============================================================================
// start runtime
//==============================================================================
void HpxTasker::startRuntime(Module &TheModule)
{
  // setup backend args
  std::vector<std::string> Argv = {"./contra"};
  
  // create constants
  auto ArgcV = llvmValue<int>(TheContext_, Argv.size());
  
  std::vector<Constant*> ArgVs;
  for (int i=0; i<Argv.size(); ++i)
    ArgVs.emplace_back( llvmString(TheContext_, TheModule, Argv[i]) );

  auto ZeroC = Constant::getNullValue(IntegerType::getInt32Ty(TheContext_));
  auto ArgvV = llvmArray(TheContext_, TheModule, ArgVs, {ZeroC, ZeroC});

  // startup runtime
  TheHelper_.callFunction(
      TheModule,
      "contra_hpx_startup",
      VoidType_,
      {ArgcV, ArgvV} );
  
  launch(TheModule, *TopLevelTask_);
}

//==============================================================================
// Launch a task
//==============================================================================
Value* HpxTasker::launch(
    Module &TheModule,
    const TaskInfo & TaskI,
    const std::vector<Value*> & ArgVs)
{
  auto F = TheModule.getFunction(TaskI.getName());

  TheHelper_.callFunction(
      TheModule,
      "contra_hpx_launch_task",
      VoidType_,
      {F});
}

} // namespace
