#include "tasking_rt.hpp"

extern "C" {
  
//==============================================================================
/// Integer max/min
//==============================================================================
int_t imax(int_t a, int_t b) 
{ return a > b ? a : b; }

int_t imin(int_t a, int_t b) 
{ return a < b ? a : b; }

} // extern
