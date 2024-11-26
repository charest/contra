#include "llvm_includes.hpp"
#include "timer.hpp"

#include "config.hpp"
#include "contra/context.hpp"
#include "contra/symbols.hpp"
#include "utils/llvm_utils.hpp"

#if defined( _WIN32 )
#include <Windows.h>
#else
#include <sys/time.h>
#endif

extern "C" {

//==============================================================================
/// generic c print statement
//==============================================================================
real_t timer(void)
{
#if defined( _WIN32 )
  
  // use windows api
  LARGE_INTEGER time,freq;
  if (!QueryPerformanceFrequency(&freq)) 
    THROW_CONTRA_ERROR( "Error getting clock frequency." );
  if (!QueryPerformanceCounter(&time))
    THROW_CONTRA_ERROR( "Error getting wall time." );
  return (real_t)time.QuadPart / freq.QuadPart;

#else
  
  // Use system time call
  struct timeval tm;
  if (gettimeofday( &tm, 0 )) THROW_CONTRA_ERROR( "Error getting wall time." );
  return (real_t)tm.tv_sec + (real_t)tm.tv_usec * 1.e-6;

#endif
}

} // extern

namespace librt {

using namespace contra;
using namespace utils;
using namespace llvm;

//==============================================================================
// Installs the print function
//==============================================================================
const std::string Timer::Name = "timer";

Function * Timer::install(LLVMContext & TheContext, Module & TheModule)
{
  auto RealType = llvmType<real_t>(TheContext);
  auto TimerType = FunctionType::get(
      RealType,
      std::nullopt,
      false /* var args */ );

  auto TimerFun = Function::Create(TimerType, Function::InternalLinkage,
      Timer::Name, TheModule);
  return TimerFun;
}


std::unique_ptr<contra::FunctionDef> Timer::check()
{
  auto & C = Context::instance();
  auto RealType = VariableType(C.getFloat64Type());
  return std::make_unique<BuiltInFunction>(
      Timer::Name, 
      RealType,
      false);
}

}
