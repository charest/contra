#include <hpx/hpx_start.hpp>
#include <hpx/iostream.hpp>

extern "C" {

//==============================================================================
// Additional startup operations
//==============================================================================
void contra_hpx_startup(int argc, char * argv[]) 
{
  std::cout << "starting up hpx" << std::endl;
  // Initialize HPX, don't run hpx_main
  hpx::start(nullptr, argc, argv);
}

//==============================================================================
// Additional shutdown operations
//==============================================================================
void contra_hpx_shutdown() 
{
  std::cout << "shutting down hpx" << std::endl;
  // hpx::finalize has to be called from the HPX runtime before hpx::stop
  hpx::apply([]() { hpx::finalize(); });
  hpx::stop();
}

//==============================================================================
// Launch task
//==============================================================================
void contra_hpx_launch_task( void(*task)(void) )
{
  std::cout << "launching task" << std::endl;
  // hpx::finalize has to be called from the HPX runtime before hpx::stop
  hpx::apply(*task);
}

}
