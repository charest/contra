#ifndef CONTRA_INPUTS_HPP
#define CONTRA_INPUTS_HPP

namespace contra {

struct InputsType {
  bool is_interactive = false;
  bool do_compile = false;
  bool is_verbose = false;
  bool has_output = false;
  bool is_debug = false;
  bool is_optimized = false;
  bool dump_ir = false;
  bool dump_dot = false;
};

} // namespace

#endif // CONTRA_INPUTS_HPP
