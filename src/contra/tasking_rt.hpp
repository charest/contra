#ifndef CONTRA_TASKING_RT_HPP
#define CONTRA_TASKING_RT_HPP

#include "config.hpp"

extern "C" {

////////////////////////////////////////////////////////////////////////////////
/// Types needed for defaullt runtime
////////////////////////////////////////////////////////////////////////////////

struct contra_index_space_t {
  int_t start;
  int_t end;
  int_t step;

  void setup(int_t s, int_t e, int_t stp=1)
  {
    start = s;
    end = e;
    step = stp;
  }

  int_t size() { return end - start; }
};

} // extern


#endif // CONTRA_TASKING_RT_HPP
